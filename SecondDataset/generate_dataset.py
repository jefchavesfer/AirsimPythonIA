"""
generate_dataset.py - randomized synthetic dataset generator for AirSim.

Design
  * AirSim runs in SimMode "ComputerVision": the observer is just a camera that is
    teleported every frame (no flight dynamics, no physics surprises).
  * The 3 target drones (and optional distractors) are ordinary actors placed in
    the Unreal level and moved with simSetObjectPose.
  * Each *sequence* is one short pass with its own random scenario: target class,
    approach geometry, aspect angle, range, FOV, time of day, weather, aiming error,
    camera roll, distractors. Frames are sampled at a low FPS (default 3) so that
    consecutive frames are not near-duplicates.
  * Labels come from simGetDetections and are filtered (min size, truncation,
    optional depth-based occlusion check via --occlusion-check). Everything is logged to metadata.csv so you can
    split BY SEQUENCE (split_by_sequence.py) and break results down by condition.

Usage
  python generate_dataset.py --discover
  python generate_dataset.py --preview --seqs-per-class 2 --bg-seqs 1
  python generate_dataset.py --map-name Coastal --seqs-per-class 100 --bg-seqs 20 --modality both
  # balanced by FRAMES: keeps running sequences of whichever class has the fewest saved frames
  # until every class has >= N frames (resumable: counts frames already in metadata.csv)
  python generate_dataset.py --map-name Coastal --frames-per-class 5000 --modality both
"""
import argparse
import csv
import fnmatch
import math
import os
import random
import time
from types import SimpleNamespace

import airsim
import cv2
import numpy as np

# ============================================================== CONFIG
CAMERA = "0"                      # front_center camera of the ComputerVision vehicle
IMAGE_TYPE = airsim.ImageType.Scene

# class_id: (class name, actor name in the Unreal Outliner, detection name pattern)
# Run `--discover` and adapt the third field if detections are not matched.
CLASSES = {
    0: ("bayraktar_tb2", "TargetBayraktar", "*TargetBayraktar*"),
    1: ("geran",         "TargetGeran",     "*TargetGeran*"),
    2: ("mq9_reaper",    "TargetReaper",    "*TargetReaper*"),
}

# Actors in the level used as unlabeled distractors (birds, light aircraft, balloons...).
DISTRACTORS = []                  # e.g. ["Distractor_Cessna", "Distractor_Bird"]

# Approximate real wingspans (m), used by --discover to sanity-check mesh scale.
REAL_SPAN_M = {0: 12.0, 1: 2.5, 2: 20.0}

# If a mesh's nose does not point along +X in Unreal, correct it here (yaw degrees).
MESH_YAW_OFFSET_DEG = {0: 0.0, 1: 0.0, 2: 0.0}

HIDE_POS = (30000.0, 30000.0, -100.0)   # parking spot (m), outside detection radius
DETECTION_RADIUS_CM = 300000            # 3000 m
ARENA_HALF_M = 400.0                    # target start XY drawn from +-this (relative to PlayerStart)
TARGET_ALT_M = (60.0, 500.0)
MIN_OBSERVER_ALT_M = 15.0
HOUR_RANGE = (6.0, 19.0)
FOV_CHOICES = [30, 40, 60, 75, 90]

# SSD compatibility: frames are stretched to SSD_INPUT x SSD_INPUT before the network sees them,
# so apparent target size is planned and filtered in THAT resolution.
SSD_INPUT = 300
MIN_SSD_PX = 12             # longest box side after resizing to SSD_INPUT (override: --min-ssd-px)

# Weather: (probability per sequence, max intensity). Fog is disabled on purpose.
WEATHER_CFG = {"Fog": (0.0, 0.0), "Rain": (0.2, 0.4), "Dust": (0.1, 0.3)}

# Label filters
MIN_VISIBLE_FRAC = 0.5      # clipped area / unclipped area (truncation)
MAX_OCCLUDED_FRAC = 0.35    # fraction of box pixels with something in front of the target
OCC_TOL_M = 25.0

SETTLE_S = 0.15             # wait after teleport so the renderer catches up
DETECT_ATTEMPTS = 3         # re-grab image+detections if the detection is missing/stale
CONSISTENCY_SLACK_PX = 4    # extra tolerance for the projection-vs-detection check
# ===============================================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def make_pose(p, pitch, roll, yaw):
    return airsim.Pose(airsim.Vector3r(p[0], p[1], p[2]), airsim.to_quaternion(pitch, roll, yaw))


def project_to_image(obs, pitch, roll, yaw, target, w, h, fov_deg):
    """Pinhole projection (AirSim NED, body x fwd / y right / z down, R = Rz(yaw) Ry(pitch) Rx(roll)).
    Returns the pixel (u, v) of a world point or None if it is behind the camera."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    R = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp,     cp * sr,                cp * cr]]
    d = [target[i] - obs[i] for i in range(3)]
    xb = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2]
    yb = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2]
    zb = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2]
    if xb <= 1.0:
        return None
    f = (w / 2) / math.tan(math.radians(fov_deg) / 2)
    return w / 2 + f * yb / xb, h / 2 + f * zb / xb


def name_to_class(name):
    n = name.lower()
    for cid, (_, _, pattern) in CLASSES.items():
        if fnmatch.fnmatch(n, pattern.lower()):
            return cid
    return None


# ------------------------------------------------------------ scene helpers
_park_warned = set()


def park(client, names):
    for i, n in enumerate(names):
        ok = client.simSetObjectPose(
            n, make_pose((HIDE_POS[0], HIDE_POS[1] + 60.0 * i, HIDE_POS[2]), 0, 0, 0), True)
        if not ok and n not in _park_warned:
            _park_warned.add(n)
            print(f"[WARN] AirSim could not move '{n}'. Set its Mobility to Movable in Unreal "
                  f"(root/mesh component -> Transform -> Mobility) and save the map.")


def resolve_actor(client, name):
    """AirSim finds actors by their internal ID name (e.g. 'TargetGeran_4') or by an
    actor Tag - NOT by the Outliner label. Try the configured name first, then any
    scene object whose name contains it."""
    cands = [name]
    try:
        cands += [n for n in client.simListSceneObjects(f".*{name}.*") if n not in cands]
    except Exception:
        pass
    for cand in cands:
        if not math.isnan(client.simGetObjectPose(cand).position.x_val):
            return cand
    return None


def check_actors(client):
    ok = True
    for cid, (cname, actor, pattern) in list(CLASSES.items()):
        found = resolve_actor(client, actor)
        if found is None:
            print(f"[ERROR] actor '{actor}' (class {cname}) not found by AirSim.")
            ok = False
        else:
            if found != actor:
                print(f"[INFO] '{actor}' resolved to AirSim name '{found}'")
            CLASSES[cid] = (cname, found, pattern)
    if not ok:
        try:
            sample = [n for n in client.simListSceneObjects(".*") if "target" in n.lower()][:10]
        except Exception:
            sample = []
        print("Scene objects containing 'target':", sample)
        print("Fix: select the actor in Unreal -> Details -> Actor -> Tags -> add a tag equal to the "
              "name in CLASSES, save the map and restart. (Or put the internal name from the "
              "Outliner tooltip 'ID Name' into CLASSES.)")
    resolved = []
    for n in DISTRACTORS:
        found = resolve_actor(client, n)
        if found is None:
            print(f"[WARN] distractor '{n}' not found; it will be ignored.")
        else:
            resolved.append(found)
    DISTRACTORS[:] = resolved
    return ok


def setup_detection(client, patterns=None):
    client.simClearDetectionMeshNames(CAMERA, IMAGE_TYPE)
    client.simSetDetectionFilterRadius(CAMERA, IMAGE_TYPE, DETECTION_RADIUS_CM)
    for p in (patterns or [c[2] for c in CLASSES.values()]):
        client.simAddDetectionFilterMeshName(CAMERA, IMAGE_TYPE, p)


def grab(client, with_depth):
    reqs = [airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False)]
    if with_depth:
        reqs.append(airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPerspective, True, False))
    rs = client.simGetImages(reqs)
    if not rs or rs[0].width == 0:
        return None, None
    r = rs[0]
    img = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3).copy()
    depth = None
    if with_depth and len(rs) > 1 and rs[1].width:
        depth = np.array(rs[1].image_data_float, dtype=np.float32).reshape(rs[1].height, rs[1].width)
    return img, depth


def discover(client, alt_m=100.0, use_depth=False):
    """Sweep each target over several ranges (camera at FOV 90, clear weather) and print, per range,
    (1) whether the mesh is VISIBLE (image differs from a background shot taken with all targets
    parked) and (2) whether AirSim DETECTS it (box size + implied width vs real wingspan).
    Rendering and detection are separate: boxes come from mesh bounds, so they can exist for
    invisible meshes. Depth images are NOT requested unless use_depth=True (they may hang on some
    setups). Saves annotated screenshots + clean crops for 40 m and 100 m."""
    if not check_actors(client):
        return
    setup_detection(client, ["*"])
    fov = 90.0
    client.simSetCameraFov(CAMERA, fov)
    try:
        client.simEnableWeather(True)
        set_weather = getattr(client, "simSetWeatherParameter", None) or client.simSetWeatherParameters
        for name in ("Fog", "Rain", "Dust", "Snow"):
            set_weather(getattr(airsim.WeatherParameter, name), 0.0)
    except Exception:
        pass
    client.simSetVehiclePose(make_pose((0, 0, -alt_m), 0, 0, 0), True)
    all_actors = [c[1] for c in CLASSES.values()]
    park(client, all_actors)
    time.sleep(3.0)
    grab(client, False)  # throwaway: let camera pose and auto-exposure settle
    bg, _ = grab(client, False)
    h, w = bg.shape[:2]
    fx = (w / 2) / math.tan(math.radians(fov) / 2)
    cy, cx = h // 2, w // 2

    def blur_gray(im):
        return cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.int16)

    bg_gray = blur_gray(bg)
    print(f"background captured ({w}x{h}); camera altitude {alt_m:.0f} m")
    for cid, (cname, actor, pattern) in CLASSES.items():
        print(f"\n== {cname} (actor {actor}) ==")
        print("  range | progress          | changed px -> visible?        | detection")
        for rng_m in (20, 40, 70, 100, 150, 250):
            print(f"  {rng_m:4d} m | ", end="", flush=True)
            park(client, all_actors)
            client.simSetObjectPose(actor, make_pose((rng_m, 0, -alt_m), 0, 0, 0), True)
            time.sleep(2.0)
            print("posed", end=" ", flush=True)
            grab(client, False)  # throwaway
            img, _ = grab(client, False)
            print("image", end=" ", flush=True)
            dets = client.simGetDetections(CAMERA, IMAGE_TYPE)
            print("dets", end=" ", flush=True)
            depth_txt = ""
            if use_depth:
                _, depth = grab(client, True)
                if depth is not None:
                    dmin = float(depth[cy - 12:cy + 13, cx - 12:cx + 13].min())
                    depth_txt = f" depth@center {dmin:.0f} m"
                print("depth", end=" ", flush=True)
            span_px = REAL_SPAN_M[cid] * fx / rng_m
            hw = int(clamp(span_px * 0.7 + 10, 14, 300))
            hh = int(max(10, hw / 3))
            diff = np.abs(blur_gray(img)[cy - hh:cy + hh + 1, cx - hw:cx + hw + 1] -
                          bg_gray[cy - hh:cy + hh + 1, cx - hw:cx + hw + 1])
            n_changed = int((diff > 20).sum())
            state = "visible" if n_changed >= 3 else "NOT visible (blocked/culled)"
            clean = img.copy()
            best, txt = None, "none"
            for d in dets:
                if actor.lower() not in d.name.lower():
                    continue
                b = d.box2D
                bw, bh = b.max.x_val - b.min.x_val, b.max.y_val - b.min.y_val
                implied = bw * rng_m / fx
                txt = (f"{bw:.0f}x{bh:.0f} px, {implied:.1f} m vs ~{REAL_SPAN_M[cid]} m "
                       f"(ratio {implied / REAL_SPAN_M[cid]:.2f})")
                cv2.rectangle(img, (int(b.min.x_val), int(b.min.y_val)), (int(b.max.x_val), int(b.max.y_val)),
                              (0, 255, 0), 1)
                best = (b, bw, bh)
            print(f"| {n_changed:6d} px -> {state:28s} | {txt}{depth_txt}")
            if rng_m in (40, 100):
                cv2.line(img, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 1)
                cv2.line(img, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 1)
                tag = f"discover_{cname}_{rng_m}m"
                cv2.imwrite(tag + ".png", img)
                if best:
                    b, bw, bh = best
                    pad = max(20, int(0.5 * max(bw, bh)))
                    x0, y0 = max(0, int(b.min.x_val) - pad), max(0, int(b.min.y_val) - pad)
                    x1, y1 = min(w, int(b.max.x_val) + pad), min(h, int(b.max.y_val) + pad)
                    crop = clean[y0:y1, x0:x1]
                    if crop.size:
                        sc = clamp(400 / max(crop.shape[:2]), 1, 8)
                        cv2.imwrite(tag + "_crop.png", cv2.resize(crop, None, fx=sc, fy=sc,
                                                                  interpolation=cv2.INTER_NEAREST))
    park(client, all_actors)
    print("\nHow to read this:\n"
          "  visible + detection       -> good.\n"
          "  visible + no detection    -> detection/filter problem for that actor.\n"
          "  NOT visible               -> terrain in front (try --discover-alt 200) or a draw/cull-distance problem.")


# ------------------------------------------------------------ randomization
def sample_scene(rng):
    weather = {"Fog": 0.0, "Rain": 0.0, "Dust": 0.0, "Snow": 0.0}
    for name, (prob, vmax) in WEATHER_CFG.items():
        if vmax > 0 and rng.random() < prob:
            weather[name] = rng.uniform(0.1 * vmax, vmax)
    return {"hour": rng.uniform(*HOUR_RANGE), "fov": rng.choice(FOV_CHOICES), "weather": weather}


def apply_scene(ctx, sc):
    c = ctx
    try:
        hh = int(sc["hour"])
        mm = int((sc["hour"] - hh) * 60)
        c.client.simSetTimeOfDay(True, start_datetime=f"2026-06-21 {hh:02d}:{mm:02d}:00",
                                 is_start_datetime_dst=False, celestial_clock_speed=1,
                                 update_interval_secs=10, move_sun=True)
    except Exception as e:
        if "tod" not in c.warned:
            print(f"[WARN] simSetTimeOfDay failed ({e}); time of day will not vary.")
            c.warned.add("tod")
    try:
        c.client.simEnableWeather(True)
        # the AirSim Python method is singular: simSetWeatherParameter
        set_weather = getattr(c.client, "simSetWeatherParameter", None) or c.client.simSetWeatherParameters
        for name, val in sc["weather"].items():
            set_weather(getattr(airsim.WeatherParameter, name), val)
    except Exception as e:
        if "wx" not in c.warned:
            print(f"[WARN] weather API failed ({e}); weather will not vary.")
            c.warned.add("wx")
    c.client.simSetCameraFov(CAMERA, float(sc["fov"]))
    time.sleep(1.0)


def sample_geometry(rng, min_px):
    """Relative observer->target geometry. az is measured from the target's nose
    (0 = observer ahead of the target, 180 = behind it), el positive = above.
    Distance is planned through the target's apparent WIDTH in SSD-input pixels (px0 -> px1),
    so every class/FOV produces targets an SSD300 can actually see."""
    def lg(a, b):
        return math.exp(rng.uniform(math.log(a), math.log(b)))

    prof = rng.choice(["approach", "head_on", "tail_chase", "orbit", "flyby"])
    el0 = rng.uniform(-25.0, 55.0)
    el1 = el0 + rng.gauss(0.0, 8.0)
    if prof == "approach":
        px0, px1 = lg(min_px, 30), lg(60, 160)
        az0 = rng.uniform(0, 360)
        az1 = az0 + rng.gauss(0, 20)
    elif prof == "head_on":
        px0, px1 = lg(min_px, 30), lg(50, 140)
        az0 = rng.gauss(0, 25)
        az1 = az0 + rng.gauss(0, 10)
    elif prof == "tail_chase":
        px0, px1 = lg(min_px, 25), lg(40, 120)
        az0 = 180 + rng.gauss(0, 25)
        az1 = az0 + rng.gauss(0, 10)
    elif prof == "orbit":
        px0 = px1 = lg(min_px, 120)
        az0 = rng.uniform(0, 360)
        az1 = az0 + rng.choice([-1, 1]) * rng.uniform(90, 270)
    else:  # flyby
        px0, px1 = lg(min_px, 40), lg(80, 180)
        az0 = rng.uniform(0, 360)
        az1 = az0 + rng.choice([-1, 1]) * rng.uniform(60, 150)
    return prof, px0, px1, az0, az1, el0, el1


def augment(img, rng):
    out = img.astype(np.float32)
    if rng.random() < 0.5:
        out = out * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    if rng.random() < 0.4:
        out += np.random.normal(0.0, rng.uniform(2, 10), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def to_ir(img, clahe):
    """Pseudo-IR: grayscale + CLAHE. NOT a thermal simulation"""
    return cv2.cvtColor(clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------ labels
def make_label(dets, w, h, depth, true_range, cid_active, expected_uv=None, min_ssd_px=MIN_SSD_PX):
    """Return (label_dict or None, reason_if_none). expected_uv = where our own kinematics say the
    target's origin projects; a detection far from it is stale or wrong and is rejected."""
    boxes = []
    for d in dets:
        if name_to_class(d.name) == cid_active:
            boxes.append((d.box2D.min.x_val, d.box2D.min.y_val, d.box2D.max.x_val, d.box2D.max.y_val))
    if not boxes:
        return None, "no_detection"
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    full = (x1 - x0) * (y1 - y0)
    if expected_uv is not None:
        dist = math.hypot((x0 + x1) / 2 - expected_uv[0], (y0 + y1) / 2 - expected_uv[1])
        if dist > 0.35 * max(x1 - x0, y1 - y0) + CONSISTENCY_SLACK_PX:
            return None, "inconsistent"
    cx0, cy0, cx1, cy1 = clamp(x0, 0, w), clamp(y0, 0, h), clamp(x1, 0, w), clamp(y1, 0, h)
    bw, bh = cx1 - cx0, cy1 - cy0
    if full <= 0 or bw <= 0 or bh <= 0:
        return None, "off_screen"
    vis = (bw * bh) / full
    if vis < MIN_VISIBLE_FRAC:
        return None, "truncated"
    ssd_w, ssd_h = bw * SSD_INPUT / w, bh * SSD_INPUT / h
    if max(ssd_w, ssd_h) < min_ssd_px:
        return None, "too_small"
    if bw > 0.98 * w and bh > 0.98 * h:
        return None, "full_frame"
    occ = 0.0
    if depth is not None:
        roi = depth[int(cy0):int(math.ceil(cy1)), int(cx0):int(math.ceil(cx1))]
        if roi.size:
            occ = float(np.mean(roi < true_range - OCC_TOL_M))
        if occ > MAX_OCCLUDED_FRAC:
            return None, "occluded"
    xc, yc = (cx0 + cx1) / 2 / w, (cy0 + cy1) / 2 / h
    nw, nh = bw / w, bh / h
    return {"line": f"{cid_active} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}",
            "box": (cx0, cy0, cx1, cy1), "nw": nw, "nh": nh, "vis": vis, "occ": occ,
            "ssd_w": ssd_w, "ssd_h": ssd_h}, ""


def save_frame(ctx, name, img, label_line):
    c = ctx
    base = os.path.splitext(name)[0]
    q = ctx.rng.randint(70, 95) if c.args.augment else 95
    for mod in c.mods:
        im = to_ir(img, c.clahe) if mod == "ir" else img
        path = os.path.join(c.out, mod, "images", name)
        if name.endswith(".jpg"):
            cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, q])
        else:
            cv2.imwrite(path, im)
        with open(os.path.join(c.out, mod, "labels", base + ".txt"), "w") as f:
            f.write(label_line)


# ------------------------------------------------------------ one sequence
def run_sequence(ctx, seq_id, cid):
    c, rng, client = ctx, ctx.rng, ctx.client
    active = cid >= 0
    cname, actor = (CLASSES[cid][0], CLASSES[cid][1]) if active else ("background", None)

    sc = sample_scene(rng)
    apply_scene(c, sc)
    span = REAL_SPAN_M[cid] if active else REAL_SPAN_M[0]
    f_ssd = (SSD_INPUT / 2) / math.tan(math.radians(sc["fov"]) / 2)  # focal length in SSD-input px
    prof, px0, px1, az0, az1, el0, el1 = sample_geometry(rng, c.args.min_ssd_px)
    dt = 1.0 / c.args.fps
    n = max(4, int(rng.uniform(10.0, 30.0) * c.args.fps))

    # target kinematics (the observer is defined relative to the target)
    v = rng.uniform(20.0, 60.0)
    psi = rng.uniform(-math.pi, math.pi)
    omega = math.radians(rng.gauss(0.0, 3.0))
    gamma = math.radians(rng.gauss(0.0, 3.0))
    pos = [rng.uniform(-ARENA_HALF_M, ARENA_HALF_M), rng.uniform(-ARENA_HALF_M, ARENA_HALF_M),
           -rng.uniform(*TARGET_ALT_M)]

    # aiming error / camera roll
    hfov = math.radians(sc["fov"])
    vfov = 2 * math.atan(math.tan(hfov / 2) * c.h / c.w)
    umax = rng.choice([0.1, 0.35, 0.7])
    u, vv = rng.uniform(-umax, umax), rng.uniform(-umax, umax)
    roll0 = math.radians(rng.uniform(-15, 15))

    all_actors = [x[1] for x in CLASSES.values()] + list(c.distractors)
    park(client, all_actors)
    dstate = []
    if c.distractors and rng.random() < 0.5:
        for name in rng.sample(c.distractors, k=min(len(c.distractors), rng.randint(1, 2))):
            dstate.append({"name": name,
                           "pos": [pos[0] + rng.uniform(-200, 200), pos[1] + rng.uniform(-200, 200),
                                   pos[2] + rng.uniform(-60, 60)],
                           "yaw": rng.uniform(-math.pi, math.pi), "v": rng.uniform(5, 40)})

    saved = 0
    for k in range(n):
        s = smoothstep(k / (n - 1))
        px_now = math.exp(math.log(px0) + (math.log(px1) - math.log(px0)) * s)
        R = clamp(span * f_ssd / px_now, 1.2 * span, 3000.0)
        az = az0 + (az1 - az0) * s
        el = el0 + (el1 - el0) * s
        ang = psi + math.radians(az)
        obs = [pos[0] + R * math.cos(math.radians(el)) * math.cos(ang),
               pos[1] + R * math.cos(math.radians(el)) * math.sin(ang),
               pos[2] - R * math.sin(math.radians(el))]
        obs[2] = min(obs[2], -MIN_OBSERVER_ALT_M)
        d = [pos[i] - obs[i] for i in range(3)]
        true_range = math.sqrt(sum(x * x for x in d))
        bearing = math.atan2(d[1], d[0])
        elev = math.atan2(-d[2], math.hypot(d[0], d[1]))

        # keep the whole target inside the frame (AirSim returns no box for partly off-screen meshes)
        ucap = min(umax, max(0.05, 0.92 - px_now / SSD_INPUT))
        vcap = min(umax, 0.75)
        u = clamp(u + rng.gauss(0, 0.08), -ucap, ucap)
        vv = clamp(vv + rng.gauss(0, 0.08), -vcap, vcap)
        cam_yaw = bearing - math.atan(u * math.tan(hfov / 2)) + math.radians(rng.gauss(0, 0.25))
        cam_pitch = elev - math.atan(vv * math.tan(vfov / 2)) + math.radians(rng.gauss(0, 0.25))
        cam_roll = roll0 + math.radians(rng.gauss(0, 0.5))
        client.simSetVehiclePose(make_pose(obs, cam_pitch, cam_roll, cam_yaw), True)

        if active:
            bank = clamp(math.atan(v * omega / 9.81), -math.radians(35), math.radians(35))
            client.simSetObjectPose(actor, make_pose(pos, gamma, bank,
                                                     psi + math.radians(MESH_YAW_OFFSET_DEG[cid])), True)
        for ds in dstate:
            client.simSetObjectPose(ds["name"], make_pose(ds["pos"], 0, 0, ds["yaw"]), True)
            ds["pos"][0] += ds["v"] * dt * math.cos(ds["yaw"])
            ds["pos"][1] += ds["v"] * dt * math.sin(ds["yaw"])

        time.sleep(SETTLE_S)
        if k == 0:  # new scene/FOV/time of day: let the renderer and auto-exposure settle
            for _ in range(2):
                time.sleep(0.3)
                grab(client, False)
        expected = (project_to_image(obs, cam_pitch, cam_roll, cam_yaw, pos, c.w, c.h, sc["fov"])
                    if active else None)
        img = depth = lab = None
        reason = "no_image"
        for attempt in range(DETECT_ATTEMPTS):
            img, depth = grab(client, active and c.args.check_occlusion)
            if img is None:
                time.sleep(0.1)
                continue
            if not active:
                break
            dets = client.simGetDetections(CAMERA, IMAGE_TYPE)
            lab, reason = make_label(dets, c.w, c.h, depth, true_range, cid, expected, c.args.min_ssd_px)
            if lab is not None or reason not in ("no_detection", "inconsistent"):
                break
            time.sleep(0.1)  # transient: detection missing or stale, try again
        if img is not None:
            fs = c.fov_stats.setdefault(sc["fov"], [0, 0])
            fs[0] += 1
            if active and lab is None:
                key = reason
                if reason == "no_detection" and expected is not None:
                    inner = 0.15 * c.w < expected[0] < 0.85 * c.w and 0.15 * c.h < expected[1] < 0.85 * c.h
                    key = "no_detection(inner)" if inner else "no_detection(edge)"
                c.skipped[key] = c.skipped.get(key, 0) + 1
                c.empty_streak += 1 if reason == "no_detection" else 0
                if c.empty_streak >= 40:
                    raise RuntimeError("40 frames in a row without any detection of the active target. "
                                       "Run `--discover` and fix the patterns in CLASSES.")
            else:
                c.empty_streak = 0
                if c.args.augment:
                    img = augment(img, rng)
                name = f"s{seq_id:05d}_f{k:04d}.{c.args.ext}"
                save_frame(c, name, img, lab["line"] if lab else "")
                c.writer.writerow({
                    "file": name, "seq": seq_id, "class_id": cid, "class_name": cname, "profile": prof,
                    "t": f"{k * dt:.2f}", "range_m": f"{true_range:.1f}", "az_deg": f"{az % 360:.1f}",
                    "el_deg": f"{el:.1f}", "target_speed": f"{v:.1f}", "fov": sc["fov"],
                    "hour": f"{sc['hour']:.2f}", "fog": f"{sc['weather']['Fog']:.2f}",
                    "rain": f"{sc['weather']['Rain']:.2f}", "dust": f"{sc['weather']['Dust']:.2f}",
                    "aim_u": f"{u:.2f}", "aim_v": f"{vv:.2f}",
                    "box_w": f"{lab['nw']:.4f}" if lab else "0", "box_h": f"{lab['nh']:.4f}" if lab else "0",
                    "vis_frac": f"{lab['vis']:.2f}" if lab else "", "occ_frac": f"{lab['occ']:.2f}" if lab else "",
                    "ssd_w": f"{lab['ssd_w']:.1f}" if lab else "0", "ssd_h": f"{lab['ssd_h']:.1f}" if lab else "0",
                    "size_bin": size_bin(lab), "px_planned": f"{px_now:.1f}",
                    "n_distractors": len(dstate), "map": c.args.map_name, "seed": c.seed})
                c.meta_file.flush()
                saved += 1
                fs[1] += 1
                c.per_class[cid] = c.per_class.get(cid, 0) + 1
                if c.args.preview:
                    vis_img = img.copy()
                    if lab:
                        x0, y0, x1, y1 = [int(t) for t in lab["box"]]
                        cv2.rectangle(vis_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(vis_img, f"{cname} ssd {lab['ssd_w']:.0f}x{lab['ssd_h']:.0f}",
                                    (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow("preview (q to quit)", vis_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

        # advance target
        psi += omega * dt
        pos[0] += v * dt * math.cos(gamma) * math.cos(psi)
        pos[1] += v * dt * math.cos(gamma) * math.sin(psi)
        pos[2] -= v * dt * math.sin(gamma)
        if not (TARGET_ALT_M[0] <= -pos[2] <= TARGET_ALT_M[1]):
            gamma = -gamma
    print(f"seq {seq_id:05d} {cname:14s} {prof:10s} hour={sc['hour']:.1f} fov={sc['fov']:3d} "
          f"fog={sc['weather']['Fog']:.2f} rain={sc['weather']['Rain']:.2f} -> {saved}/{n} frames")


# ------------------------------------------------------------ main
META_FIELDS = ["file", "seq", "class_id", "class_name", "profile", "t", "range_m", "az_deg", "el_deg",
               "target_speed", "fov", "hour", "fog", "rain", "dust", "aim_u", "aim_v", "box_w", "box_h",
               "vis_frac", "occ_frac", "ssd_w", "ssd_h", "size_bin", "px_planned", "n_distractors", "map", "seed"]


def size_bin(lab):
    """COCO-style size bin of the box in SSD-input pixels (sqrt of area)."""
    if not lab:
        return "none"
    side = math.sqrt(lab["ssd_w"] * lab["ssd_h"])
    return "small" if side < 32 else "medium" if side < 96 else "large"


def count_frames(path):
    """Saved frames per class_id already present in metadata.csv (-1 = background)."""
    counts = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                k = int(r["class_id"])
                counts[k] = counts.get(k, 0) + 1
    return counts


def next_class(per_class, target, bg_frac, rng, fails):
    """Frame-balanced scheduler. Returns the class to run next, or None when done.
    Background sequences are interleaved so they stay ~bg_frac of all frames."""
    need = [cid for cid in CLASSES if per_class.get(cid, 0) < target and fails.get(cid, 0) < 15]
    n_target = sum(per_class.get(cid, 0) for cid in CLASSES)
    n_bg = per_class.get(-1, 0)
    bg_goal = bg_frac / (1 - bg_frac) * (n_target if need else len(CLASSES) * target)
    if n_bg < bg_goal and (n_bg < 0.8 * bg_goal or not need):
        return -1
    if not need:
        return None
    low = min(per_class.get(cid, 0) for cid in need)
    return rng.choice([cid for cid in need if per_class.get(cid, 0) == low])


def last_seq(path):
    if not os.path.exists(path):
        return -1
    with open(path, newline="") as f:
        return max((int(r["seq"]) for r in csv.DictReader(f)), default=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--map-name", default="unknown", help="label stored in metadata (use one per Unreal map)")
    ap.add_argument("--seqs-per-class", type=int, default=100)
    ap.add_argument("--bg-seqs", type=int, default=20, help="sequences without any target (empty labels)")
    ap.add_argument("--frames-per-class", type=int, default=0,
                    help="balance by FRAMES: keep running sequences of the class with the fewest saved frames "
                         "until each class has this many (counts frames already in the dataset; overrides "
                         "--seqs-per-class/--bg-seqs). Each class can overshoot by up to one sequence.")
    ap.add_argument("--bg-frac", type=float, default=0.07,
                    help="with --frames-per-class: background frames as a fraction of all frames")
    ap.add_argument("--max-seqs", type=int, default=5000, help="safety cap on sequences in this run")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--modality", choices=["rgb", "ir", "both"], default="rgb")
    ap.add_argument("--ext", choices=["jpg", "png"], default="png",
                    help="png matches the earlier pipeline; jpg saves a lot of disk space")
    ap.add_argument("--min-ssd-px", type=int, default=MIN_SSD_PX,
                    help="drop labels whose longest side is below this after resizing to SSD input (300)")
    ap.add_argument("--augment", action="store_true", help="photometric noise/blur/brightness + random JPEG quality")
    ap.add_argument("--occlusion-check", dest="check_occlusion", action="store_true",
                    help="request depth images to reject occluded targets (off by default; may hang on some setups)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--discover-alt", type=float, default=100.0, help="camera altitude (m) for --discover")
    ap.add_argument("--discover-depth", action="store_true", help="also request depth images in --discover")
    args = ap.parse_args()

    client = airsim.VehicleClient(timeout_value=60)   # a hung call raises after 60 s instead of freezing
    client.confirmConnection()
    if args.discover:
        discover(client, args.discover_alt, args.discover_depth)
        return
    if not check_actors(client):
        return

    seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000
    rng = random.Random(seed)
    np.random.seed(seed)
    print(f"seed = {seed}")

    mods = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in mods:
        os.makedirs(os.path.join(args.out, m, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, m, "labels"), exist_ok=True)

    meta_path = os.path.join(args.out, "metadata.csv")
    if os.path.exists(meta_path):
        with open(meta_path, newline="") as f:
            old_header = next(csv.reader(f), [])
        if old_header != META_FIELDS:
            raise SystemExit(f"{meta_path} was written by an older version of this script. "
                             f"Use a fresh --out directory (or delete the old one).")
    start = last_seq(meta_path) + 1
    new_file = not os.path.exists(meta_path)
    meta_file = open(meta_path, "a", newline="")
    writer = csv.DictWriter(meta_file, fieldnames=META_FIELDS)
    if new_file:
        writer.writeheader()

    setup_detection(client)
    probe, _ = grab(client, False)
    if probe is None:
        raise RuntimeError("Could not grab an image. Check the camera name and settings.json.")
    ctx = SimpleNamespace(client=client, rng=rng, args=args, mods=mods, out=args.out,
                          h=probe.shape[0], w=probe.shape[1], writer=writer, meta_file=meta_file,
                          clahe=cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)), seed=seed,
                          distractors=[d for d in DISTRACTORS
                                       if not math.isnan(client.simGetObjectPose(d).position.x_val)],
                          warned=set(), skipped={}, per_class=count_frames(meta_path), empty_streak=0, fov_stats={})

    try:
        if args.frames_per_class > 0:
            print(f"Balanced mode: target {args.frames_per_class} frames per class; already in dataset: "
                  f"{dict(sorted(ctx.per_class.items()))}")
            fails, i = {}, 0
            while i < args.max_seqs:
                cid = next_class(ctx.per_class, args.frames_per_class, args.bg_frac, rng, fails)
                if cid is None:
                    break
                before = ctx.per_class.get(cid, 0)
                run_sequence(ctx, start + i, cid)
                i += 1
                if cid >= 0:
                    fails[cid] = 0 if ctx.per_class.get(cid, 0) > before else fails.get(cid, 0) + 1
                if i % 10 == 0:
                    print(f"  progress: {dict(sorted(ctx.per_class.items()))} / target {args.frames_per_class}")
            stuck = [CLASSES[c][0] for c, n in fails.items() if n >= 15]
            if stuck:
                print(f"[WARN] gave up on {stuck}: 15 sequences in a row produced no frames.")
        else:
            schedule = [cid for cid in CLASSES for _ in range(args.seqs_per_class)] + [-1] * args.bg_seqs
            rng.shuffle(schedule)
            for i, cid in enumerate(schedule):
                run_sequence(ctx, start + i, cid)
    except KeyboardInterrupt:
        print("\nInterrupted; data written so far is kept (re-run to append).")
    finally:
        park(client, [x[1] for x in CLASSES.values()] + ctx.distractors)
        meta_file.close()
        cv2.destroyAllWindows()
        print("\nFrames per class now in dataset (-1 = background):", dict(sorted(ctx.per_class.items())))
        print("Frames skipped by reason:", ctx.skipped)
        print("Frames saved / attempted by FOV:",
              {k: f"{v[1]}/{v[0]}" for k, v in sorted(ctx.fov_stats.items())})


if __name__ == "__main__":
    main()
