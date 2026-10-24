extends Node2D
## Smart Traffic Junction - one junction, four signals.
##
## The game moves the vehicles. Every simulated second it sends the junction
## state to the Python backend, which runs the chosen algorithm (plus the
## ambulance rule) and answers "which road gets green". If the backend is not
## running the game falls back to a simple fixed timer so it never freezes.
##
## Traffic drives on the LEFT (India). Only one road is green at a time, so
## every movement (straight, left, right) is conflict-free.

const VehicleScript := preload("res://scripts/vehicle.gd")
const ChartScript := preload("res://scripts/chart.gd")
const OverlayScript := preload("res://scripts/overlay.gd")

const BACKEND := "http://127.0.0.1:8765"
const PX := 6.0                       # pixels per metre
const C := Vector2(790, 450)          # junction centre on screen
const LW := 34.0                      # lane width (px); one lane each way
const LO := 17.0                      # lane centre offset from road centre
const BOX := 38.0                     # half size of the junction box
const STOP := 64.0                    # stop line distance from centre
const ROAD_LEN := 450.0               # spawn point distance from centre
const EXIT_LEN := 560.0
const B_COMF := 4.5 * PX              # comfortable braking (px/s^2)
const B_MAX := 9.0 * PX               # emergency braking
const MIN_GAP := 2.0 * PX
const SENSOR_LEN := 30.0 * PX         # actuated loop-sensor zone length
const YELLOW_T := 3.0
const ALLRED_T := 2.0
const MIN_GREEN := 8.0
const FIXED_GREEN := 20.0
const EV_RANGE_M := 120.0

const NAMES := ["North", "East", "South", "West"]
const DIR_COLORS := [Color("3b82f6"), Color("f59e0b"), Color("10b981"), Color("a855f7")]
const HEADINGS := [Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1), Vector2(1, 0)]
const ALGOS := ["fixed", "actuated", "max_pressure", "rl"]
const ALGO_NAMES := ["Fixed timer", "Actuated (sensor)", "Max-pressure", "RL agent (PPO)"]
const ALGO_TEXT := [
	"Gives each road 20 s of green in a fixed cycle N > E > S > W, whether or not cars are waiting. This is how most old signals work.",
	"A loop sensor (cyan zone) sits before each stop line. Green is extended while vehicles keep crossing it; when there is a gap, the next road that HAS cars gets green.",
	"Gives green to the road with the most 'pressure' (cars waiting to enter minus cars on the exit road). It only switches when another road is clearly ahead, because each switch wastes about 6 s.",
	"A neural network trained in the SUMO simulator by trial and error (PPO). It looks at queues, vehicle counts, ambulance flags and green time, and picks the road.",
]
const MIX := {"car": 0.50, "bike": 0.25, "auto": 0.12, "bus": 0.06, "truck": 0.07}
const TYPES := ["car", "bike", "auto", "bus", "truck", "ambulance"]
const TYPE_LABELS := ["Car", "Bike", "Auto", "Bus", "Truck", "Ambulance"]

# ---------------------------------------------------------------- scenario
var initial := [6, 3, 6, 3]
var flows := [6.0, 3.0, 6.0, 3.0]          # vehicles per minute
var random_amb := true
var ev_priority := true
var seed_val := 7
var algo_idx := 2
var speed := 1.0
var extra := []                             # vehicles added with the buttons (for Compare)

# ---------------------------------------------------------------- simulation
var rng := RandomNumberGenerator.new()
var sim_t := 0.0
var running := false
var lanes_in := [[], [], [], []]
var lanes_out := [[], [], [], []]
var backlog := [[], [], [], []]             # vehicles waiting to enter the screen
var next_arrival := [0.0, 0.0, 0.0, 0.0]
var next_amb := 40.0
var next_id := 0
var vehicles_layer: Node2D
var overlay: Node2D

# ---------------------------------------------------------------- signals
var current := 0
var mode := "green"                         # green | yellow | allred
var mode_t := 0.0
var green_t := 0.0
var target := 0
var decide_acc := 0.0
var reason := "Press START to begin."
var reason_source := "info"
var decisions_log := []
var ev_banner_t := -100.0
var ev_banner_dir := 0

# ---------------------------------------------------------------- stats
var completed_pending := []
var passed := 0
var wait_sum := 0.0
var wait_n := 0
var amb_wait_sum := 0.0
var amb_n := 0
var max_queue := 0
var history := [[], [], [], []]
var last_state := {}

# ---------------------------------------------------------------- backend
var online := false
var rl_ok := false
var req_pending := false
var req_sent_ms := 0
var run_started := false
var run_registered := false          # has the backend been told about this run?
var run_ack := false                 # ... and has it confirmed?
var run_req_kind := ""
var http_decide: HTTPRequest
var http_run: HTTPRequest
var http_status: HTTPRequest
var http_compare: HTTPRequest
var status_acc := 10.0

# ---------------------------------------------------------------- ui refs
var ui := {}


func _ready() -> void:
	vehicles_layer = Node2D.new()
	add_child(vehicles_layer)
	var ov := OverlayScript.new()
	ov.main = self
	overlay = ov
	add_child(overlay)
	http_decide = _make_http(_on_decide_done)
	http_run = _make_http(_on_run_done)
	http_status = _make_http(_on_status_done)
	http_compare = _make_http(_on_compare_done)
	_build_ui()
	reset_sim()
	_apply_cmdline()


var _shot_path := ""
var _shot_at := -1.0
var _quit_at := -1.0


func _apply_cmdline() -> void:
	## Optional launch options (handy for demos/tests), e.g.
	##   godot --path game -- --algo=2 --speed=4 --autostart
	for a in OS.get_cmdline_user_args():
		var kv: PackedStringArray = a.trim_prefix("--").split("=")
		match kv[0]:
			"algo":
				algo_idx = int(kv[1])
				ui["algo"].selected = algo_idx
				ui["how"].text = ALGO_TEXT[algo_idx]
				reset_sim()
			"speed":
				_on_speed(int(kv[1]))
			"seed":
				seed_val = int(kv[1])
				reset_sim()
			"ambulance":
				add_vehicles(int(kv[1]), "ambulance", 1)
			"autostart":
				_on_start()
			"screenshot":
				_shot_path = kv[1]
			"shot_at":
				_shot_at = float(kv[1])
			"quit_at":
				_quit_at = float(kv[1])


func _test_hooks() -> void:
	if _shot_path != "" and _shot_at >= 0.0 and sim_t >= _shot_at:
		_shot_at = -1.0
		await RenderingServer.frame_post_draw
		get_viewport().get_texture().get_image().save_png(_shot_path)
		print("screenshot saved t=%.0f passed=%d avg_wait=%.1f amb=%d online=%s" % [sim_t, passed, wait_sum / maxf(wait_n, 1), amb_n, online])
	if _quit_at >= 0.0 and sim_t >= _quit_at:
		print("quit t=%.0f passed=%d avg_wait=%.1f amb_wait=%.1f max_queue=%d online=%s" % [sim_t, passed, wait_sum / maxf(wait_n, 1), amb_wait_sum / maxf(amb_n, 1), max_queue, online])
		get_tree().quit()


func _make_http(cb: Callable) -> HTTPRequest:
	var h := HTTPRequest.new()
	h.timeout = 30.0
	add_child(h)
	h.request_completed.connect(cb)
	return h


# =================================================================== loop
func _process(delta: float) -> void:
	status_acc += delta
	if not online and status_acc > 3.0:
		status_acc = 0.0
		if http_status.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
			http_status.request(BACKEND + "/api/status")
	if online and req_pending and Time.get_ticks_msec() - req_sent_ms > 4000:
		http_decide.cancel_request()
		req_pending = false
		_set_online(false)

	if running:
		var total := delta * speed
		while total > 0.0:
			var dt := minf(total, 0.05)
			_step(dt)
			total -= dt
	for veh in vehicles_layer.get_children():
		veh.place()
	queue_redraw()
	overlay.queue_redraw()
	_update_panel()
	_test_hooks()


func _step(dt: float) -> void:
	sim_t += dt
	_arrivals()
	_spawn()
	_signal_tick(dt)
	for k in 4:
		var lane: Array = lanes_in[k]
		for i in lane.size():
			_drive(lane[i], lane[i - 1] if i > 0 else null, false, dt)
		var out: Array = lanes_out[k]
		for i in out.size():
			_drive(out[i], out[i - 1] if i > 0 else null, true, dt)
	# move vehicles that left the junction box onto their exit road
	for k in 4:
		var lane: Array = lanes_in[k]
		while not lane.is_empty() and lane[0].s > lane[0].jend_s:
			var veh = lane.pop_front()
			lanes_out[veh.exit_idx].append(veh)
		var out: Array = lanes_out[k]
		while not out.is_empty() and out[0].s >= out[0].path_len - 2.0:
			_finish(out.pop_front())
	decide_acc += dt
	if decide_acc >= 1.0:
		decide_acc -= 1.0
		last_state = _build_state()
		_record_history(last_state)
		if mode == "green":
			_decide(last_state)


# ------------------------------------------------------------------ traffic
func _exp(rate_per_min: float) -> float:
	if rate_per_min <= 0.0:
		return 1e12
	return -log(1.0 - rng.randf()) / (rate_per_min / 60.0)


func _pick_type() -> String:
	var r := rng.randf()
	var acc := 0.0
	for t in MIX:
		acc += MIX[t]
		if r <= acc:
			return t
	return "car"


func _arrivals() -> void:
	for k in 4:
		while sim_t >= next_arrival[k]:
			backlog[k].append({"type": _pick_type(), "t": next_arrival[k]})
			next_arrival[k] += _exp(flows[k])
	if random_amb and sim_t >= next_amb:
		backlog[rng.randi_range(0, 3)].append({"type": "ambulance", "t": sim_t})
		next_amb += rng.randf_range(45.0, 90.0)


func _entry_free(k: int) -> bool:
	var lane: Array = lanes_in[k]
	if lane.is_empty():
		return true
	var last = lane[lane.size() - 1]
	return last.s - last.length > MIN_GAP + 4.0


func _spawn() -> void:
	for k in 4:
		if not backlog[k].is_empty() and _entry_free(k):
			var b: Dictionary = backlog[k].pop_front()
			var veh = _new_vehicle(k, b["type"], b["t"])
			veh.wait = maxf(0.0, sim_t - b["t"] - 0.5)
			veh.v = veh.vmax * 0.6
			lanes_in[k].append(veh)


func _new_vehicle(k: int, vtype: String, t0: float):
	var veh = VehicleScript.new()
	veh.setup(vtype, rng)
	veh.id = next_id
	next_id += 1
	veh.dir = k
	veh.spawn_t = t0
	var r := rng.randf()
	veh.turn = 0 if r < 0.6 else (1 if r < 0.8 else 2)
	_build_path(veh)
	vehicles_layer.add_child(veh)
	return veh


func _left(h: Vector2) -> Vector2:
	return Vector2(h.y, -h.x)       # left-hand side of a driver heading h (screen coords)


func _build_path(veh) -> void:
	var h: Vector2 = HEADINGS[veh.dir]
	var lf := _left(h)
	var h2 := h
	if veh.turn == 1:
		h2 = lf                    # left turn (short turn in left-hand traffic)
	elif veh.turn == 2:
		h2 = -lf                   # right turn (crosses the junction)
	var lf2 := _left(h2)
	var entry := C - h * ROAD_LEN + lf * LO
	var p1 := C - h * BOX + lf * LO
	var p2 := C + h2 * BOX + lf2 * LO
	var ex := C + h2 * EXIT_LEN + lf2 * LO
	var cv := Curve2D.new()
	cv.bake_interval = 3.0
	cv.add_point(entry)
	if veh.turn != 0:
		var cc := p1 + h * (p2 - p1).dot(h)
		cv.add_point(p1, Vector2.ZERO, (cc - p1) * 2.0 / 3.0)
		cv.add_point(p2, (cc - p2) * 2.0 / 3.0, Vector2.ZERO)
	cv.add_point(ex)
	veh.curve = cv
	veh.path_len = cv.get_baked_length()
	veh.stop_s = ROAD_LEN - STOP
	veh.jstart_s = ROAD_LEN - BOX
	veh.jend_s = veh.path_len - (EXIT_LEN - BOX)
	for i in 4:
		if HEADINGS[i] == -h2:
			veh.exit_idx = i


func _must_stop(k: int, d: float, v: float) -> bool:
	if d < -0.5:
		return false
	if k == current and mode == "green":
		return false
	if k == current and mode == "yellow":
		return d > v * v / (2.0 * B_COMF) + 3.0     # can stop comfortably -> stop
	return d >= v * v / (2.0 * B_MAX)                # red: stop unless physically impossible


func _drive(veh, leader, on_exit: bool, dt: float) -> void:
	var tv: float = veh.vmax
	if veh.turn != 0 and veh.s > veh.jstart_s - 25.0 and veh.s < veh.jend_s:
		tv = minf(tv, 7.0 * PX)
	var gap := 1e9
	if leader != null:
		if on_exit:
			gap = (leader.s - leader.jend_s) - (veh.s - veh.jend_s) - leader.length - MIN_GAP
		else:
			gap = leader.s - leader.length - veh.s - MIN_GAP
	if not veh.passed_line:
		var d: float = veh.stop_s - veh.s
		if _must_stop(veh.dir, d, veh.v):
			gap = minf(gap, d - 1.0)
	tv = minf(tv, sqrt(2.0 * B_COMF * maxf(gap, 0.0)))
	var old_v: float = veh.v
	if veh.v < tv:
		veh.v = minf(tv, veh.v + veh.accel * dt)
	else:
		veh.v = maxf(tv, veh.v - B_MAX * dt)
	var move: float = minf(veh.v * dt, maxf(gap, 0.0))
	if move < veh.v * dt:
		veh.v = move / dt
	veh.s += move
	veh.braking = veh.v < old_v - 0.01
	if veh.v < 1.0 * PX:
		veh.wait += dt
	if not veh.passed_line and veh.s >= veh.stop_s:
		veh.passed_line = true


func _finish(veh) -> void:
	var rec := {"id": veh.id, "type": veh.vtype, "dir": veh.dir, "turn": ["straight", "left", "right"][veh.turn],
		"spawn_t": snappedf(veh.spawn_t, 0.1), "exit_t": snappedf(sim_t, 0.1),
		"wait": snappedf(veh.wait, 0.01), "travel": snappedf(sim_t - veh.spawn_t, 0.01)}
	completed_pending.append(rec)
	passed += 1
	if veh.vtype == "ambulance":
		amb_wait_sum += veh.wait
		amb_n += 1
	else:
		wait_sum += veh.wait
		wait_n += 1
	veh.queue_free()


# ------------------------------------------------------------------ signals
func _signal_tick(dt: float) -> void:
	mode_t += dt
	if mode == "green":
		green_t += dt
	elif mode == "yellow" and mode_t >= YELLOW_T:
		mode = "allred"
		mode_t = 0.0
	elif mode == "allred" and mode_t >= ALLRED_T:
		current = target
		mode = "green"
		mode_t = 0.0
		green_t = 0.0


func _light(k: int) -> String:
	if k != current:
		return "red"
	if mode == "green":
		return "green"
	if mode == "yellow":
		return "yellow"
	return "red"


func _build_state() -> Dictionary:
	var q := [0, 0, 0, 0]
	var ap := [0, 0, 0, 0]
	var det := [0, 0, 0, 0]
	var ev := [-1.0, -1.0, -1.0, -1.0]
	for k in 4:
		for veh in lanes_in[k]:
			if veh.passed_line:
				if veh.s - veh.stop_s < 15.0 and veh.v > 1.5 * PX:
					det[k] += 1
				continue
			ap[k] += 1
			var d: float = veh.stop_s - veh.s
			if veh.v < 1.5 * PX:
				q[k] += 1
			elif d < SENSOR_LEN:
				det[k] += 1
			if veh.vtype == "ambulance":
				var dm := d / PX
				ev[k] = dm if ev[k] < 0.0 else minf(ev[k], dm)
		for b in backlog[k]:
			ap[k] += 1
			q[k] += 1
			if b["type"] == "ambulance" and ev[k] < 0.0:
				ev[k] = (ROAD_LEN - STOP) / PX + 10.0
	var tot := 0
	for k in 4:
		tot += q[k]
	max_queue = maxi(max_queue, tot)
	return {"t": snappedf(sim_t, 0.1), "current": current, "green_time": snappedf(green_t, 0.1),
		"queue": q, "approaching": ap, "detector": det, "ev_dist": ev}


func _record_history(st: Dictionary) -> void:
	for k in 4:
		history[k].append(st["queue"][k])
		if history[k].size() > 180:
			history[k].pop_front()


func _decide(st: Dictionary) -> void:
	if online:
		if req_pending:
			return
		if not run_ack:
			if not run_registered:
				_start_run()
			return                       # wait (1 s at most) until the backend has opened the run
		var payload := {"algo": ALGOS[algo_idx], "ev_priority": ev_priority, "state": st,
			"completed": completed_pending}
		completed_pending = []
		var err := http_decide.request(BACKEND + "/api/decide", ["Content-Type: application/json"],
			HTTPClient.METHOD_POST, JSON.stringify(payload))
		if err == OK:
			req_pending = true
			req_sent_ms = Time.get_ticks_msec()
			return
		_set_online(false)
	var r := _local_decide(st)
	_apply_decision(r[0], r[1], r[2])


func _local_decide(st: Dictionary) -> Array:
	## Fallback when the backend is not running: ambulance rule + fixed timer.
	if ev_priority:
		var best := -1
		for k in 4:
			var d: float = st["ev_dist"][k]
			if d >= 0.0 and d <= EV_RANGE_M and (best < 0 or d < st["ev_dist"][best]):
				best = k
		if best >= 0:
			if best == current:
				return [current, "AMBULANCE on %s -> hold green until it passes" % NAMES[best], "emergency"]
			return [best, "AMBULANCE approaching from %s -> switch to green for it" % NAMES[best], "emergency"]
	if green_t < MIN_GREEN:
		return [current, "minimum green (8 s) not reached yet", "min_green"]
	if green_t >= FIXED_GREEN:
		var nxt := (current + 1) % 4
		return [nxt, "OFFLINE fixed timer: next in the cycle is %s" % NAMES[nxt], "algorithm"]
	return [current, "OFFLINE fixed timer (start backend/server.py for the real algorithms)", "algorithm"]


func _apply_decision(phase: int, why: String, src: String) -> void:
	reason = why
	reason_source = src
	if src == "emergency":
		ev_banner_t = sim_t
		ev_banner_dir = phase
	if mode == "green" and phase != current:
		target = phase
		mode = "yellow"
		mode_t = 0.0
		decisions_log.push_front({"t": sim_t, "text": "%s -> %s: %s" % [NAMES[current], NAMES[phase], why], "src": src})
		if decisions_log.size() > 6:
			decisions_log.pop_back()


func _on_decide_done(result: int, code: int, _h: PackedStringArray, body: PackedByteArray) -> void:
	req_pending = false
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		_set_online(false)
		return
	var d = JSON.parse_string(body.get_string_from_utf8())
	if d == null or not (d is Dictionary):
		return
	_apply_decision(int(d["phase"]), str(d["reason"]), str(d["source"]))


func _on_status_done(result: int, code: int, _h: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		_set_online(false)
		return
	var d = JSON.parse_string(body.get_string_from_utf8())
	rl_ok = d is Dictionary and bool(d.get("rl_available", false))
	_set_online(true)
	if run_started and not run_registered:
		_start_run()


func _set_online(v: bool) -> void:
	online = v
	if ui.has("algo"):
		ui["algo"].set_item_disabled(3, not (online and rl_ok))
		ui["backend"].text = "Backend: connected  (graphs at %s)" % BACKEND if online else "Backend: OFFLINE - run  python backend/server.py\n(the game uses a plain fixed timer until then)"
		ui["backend"].add_theme_color_override("font_color", Color("34d399") if online else Color("f87171"))


func _scenario() -> Dictionary:
	return {"initial": initial.duplicate(), "flows": flows.duplicate(), "random_ambulances": random_amb,
		"seed": seed_val, "extra": extra.duplicate(), "mix": MIX}


func _start_run() -> void:
	run_started = true
	if not online:
		return
	run_registered = true
	run_ack = false
	run_req_kind = "start"
	if http_run.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		http_run.cancel_request()
	http_run.request(BACKEND + "/api/start", ["Content-Type: application/json"], HTTPClient.METHOD_POST,
		JSON.stringify({"algo": ALGOS[algo_idx], "scenario": _scenario(), "ev_priority": ev_priority}))


func _stop_run() -> void:
	if online and run_registered and http_run.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
		run_req_kind = "stop"
		http_run.request(BACKEND + "/api/stop", ["Content-Type: application/json"], HTTPClient.METHOD_POST, "{}")


func _on_run_done(result: int, code: int, _h: PackedStringArray, _b: PackedByteArray) -> void:
	if run_req_kind == "start":
		if result == HTTPRequest.RESULT_SUCCESS and code == 200:
			run_ack = true
		else:
			run_registered = false
			_set_online(false)
	run_req_kind = ""


# ------------------------------------------------------------------ reset
func reset_sim() -> void:
	running = false
	for veh in vehicles_layer.get_children():
		veh.queue_free()
	lanes_in = [[], [], [], []]
	lanes_out = [[], [], [], []]
	backlog = [[], [], [], []]
	history = [[], [], [], []]
	rng.seed = seed_val
	sim_t = 0.0
	current = 0
	mode = "green"
	mode_t = 0.0
	green_t = 0.0
	decide_acc = 0.0
	reason = "Press START. Set the traffic on the left first if you like."
	reason_source = "info"
	decisions_log = []
	completed_pending = []
	passed = 0
	wait_sum = 0.0
	wait_n = 0
	amb_wait_sum = 0.0
	amb_n = 0
	max_queue = 0
	extra = []
	run_started = false
	run_registered = false
	run_ack = false
	ev_banner_t = -100.0
	next_id = 0
	for k in 4:
		next_arrival[k] = _exp(flows[k])
	next_amb = rng.randf_range(20.0, 50.0)
	# vehicles already waiting at the start: put them in the queue
	for k in 4:
		var s := ROAD_LEN - STOP - 1.0
		for i in initial[k]:
			var t := _pick_type()
			var veh = _new_vehicle(k, t, 0.0)
			if s - veh.length < 0.0:
				vehicles_layer.remove_child(veh)
				veh.free()
				backlog[k].append({"type": t, "t": 0.0})
				continue
			veh.s = s
			veh.v = 0.0
			lanes_in[k].append(veh)
			s -= veh.length + MIN_GAP + 1.0
	last_state = _build_state()
	_update_start_button()


func _add_random(k: int, count: int) -> void:
	for i in count:
		add_vehicles(k, _pick_type(), 1)


func add_vehicles(k: int, vtype: String, count: int) -> void:
	for i in count:
		backlog[k].append({"type": vtype, "t": sim_t})
		extra.append({"t": snappedf(sim_t, 0.1), "dir": k, "type": vtype})
	if not running:
		_spawn()
		last_state = _build_state()


# =================================================================== drawing
func _draw() -> void:
	var vp := get_viewport_rect().size
	draw_rect(Rect2(Vector2.ZERO, vp), Color("1a2a20"))
	# city blocks
	var blk := Color("22352a")
	var m := LW + 40.0
	for q in [Vector2(-1, -1), Vector2(1, -1), Vector2(-1, 1), Vector2(1, 1)]:
		var x0: float = C.x + (m if q.x > 0 else -m - 420.0)
		var y0: float = C.y + (m if q.y > 0 else -m - 420.0)
		draw_rect(Rect2(x0, y0, 420, 420), blk)
		for i in 3:
			for j in 3:
				var bx := x0 + 20 + i * 135
				var by := y0 + 20 + j * 135
				draw_rect(Rect2(bx, by, 110, 110), Color("2b4234"))
				draw_rect(Rect2(bx + 8, by + 8, 94, 94), Color("324d3c"))
	# sidewalks
	var sw := Color("4b5563")
	draw_rect(Rect2(C.x - LW - 8, 0, 2 * LW + 16, vp.y), sw)
	draw_rect(Rect2(0, C.y - LW - 8, vp.x, 2 * LW + 16), sw)
	# roads
	var asphalt := Color("2f343b")
	draw_rect(Rect2(C.x - LW, 0, 2 * LW, vp.y), asphalt)
	draw_rect(Rect2(0, C.y - LW, vp.x, 2 * LW), asphalt)
	# lane highlight for the road that is green / yellow
	for k in 4:
		var h: Vector2 = HEADINGS[k]
		var lf := _left(h)
		var lt := _light(k)
		if lt == "red":
			continue
		var col := Color(0.2, 0.9, 0.4, 0.13) if lt == "green" else Color(1, 0.75, 0.1, 0.16)
		var a := C - h * STOP
		var b := C - h * ROAD_LEN
		_draw_lane_rect(a, b, lf, col)
	# actuated: show the loop sensors
	if ALGOS[algo_idx] == "actuated":
		for k in 4:
			var h: Vector2 = HEADINGS[k]
			_draw_lane_rect(C - h * STOP, C - h * (STOP + SENSOR_LEN), _left(h), Color(0.2, 0.85, 0.95, 0.14))
	# centre lines (dashed)
	var dash := Color(0.95, 0.95, 0.95, 0.55)
	var y := 0.0
	while y < vp.y:
		if absf(y + 10 - C.y) > STOP + 4:
			draw_line(Vector2(C.x, y), Vector2(C.x, y + 18), dash, 2.0)
		y += 34
	var x := 0.0
	while x < vp.x:
		if absf(x + 10 - C.x) > STOP + 4:
			draw_line(Vector2(x, C.y), Vector2(x + 18, C.y), dash, 2.0)
		x += 34
	# zebra crossings + stop lines
	for k in 4:
		var h: Vector2 = HEADINGS[k]
		var lf := _left(h)
		var n := 7
		for i in n:
			var off := -LW + 3 + i * (2 * LW - 6) / n
			var p := C - h * (BOX + 3) + lf * (off + 3)
			var q2 := C - h * (STOP - 5) + lf * (off + 3)
			draw_line(p, q2, Color(0.92, 0.92, 0.92, 0.75), 5.0)
		draw_line(C - h * STOP, C - h * STOP + lf * LW, Color.WHITE, 3.0)


func _draw_lane_rect(a: Vector2, b: Vector2, lf: Vector2, col: Color) -> void:
	var pts := PackedVector2Array([a, b, b + lf * LW, a + lf * LW])
	draw_colored_polygon(pts, col)


func draw_overlay(o: Node2D) -> void:
	var font := ThemeDB.fallback_font
	for k in 4:
		var h: Vector2 = HEADINGS[k]
		var lf := _left(h)
		# signal head on the left kerb, just before the stop line
		var p := C - h * (STOP + 14) + lf * (LW + 20)
		o.draw_rect(Rect2(p.x - 9, p.y - 25, 18, 50), Color("0b0f14"))
		o.draw_rect(Rect2(p.x - 9, p.y - 25, 18, 50), Color("475569"), false, 1.5)
		var lt := _light(k)
		o.draw_circle(p + Vector2(0, -14), 6, Color("ef4444") if lt == "red" else Color("3b1212"))
		o.draw_circle(p, 6, Color("f59e0b") if lt == "yellow" else Color("3a2a0a"))
		o.draw_circle(p + Vector2(0, 14), 6, Color("22c55e") if lt == "green" else Color("0f2e1a"))
		if lt == "green":
			o.draw_circle(p + Vector2(0, 14), 11, Color(0.13, 0.77, 0.37, 0.25))
		# road label + queue
		var lp := C - h * 250 + lf * (LW + 62)
		var q: int = last_state.get("queue", [0, 0, 0, 0])[k] if last_state.has("queue") else 0
		o.draw_rect(Rect2(lp.x - 62, lp.y - 26, 124, 50), Color(0.04, 0.07, 0.12, 0.82))
		o.draw_rect(Rect2(lp.x - 62, lp.y - 26, 4, 50), DIR_COLORS[k])
		o.draw_string(font, Vector2(lp.x - 52, lp.y - 7), NAMES[k].to_upper(), HORIZONTAL_ALIGNMENT_LEFT, 110, 14, Color.WHITE)
		o.draw_string(font, Vector2(lp.x - 52, lp.y + 12), "stopped: %d" % q, HORIZONTAL_ALIGNMENT_LEFT, 110, 13, Color("cbd5e1"))
		if not backlog[k].is_empty():
			var ep := C - h * (ROAD_LEN - 40) + lf * (LW + 36)
			o.draw_rect(Rect2(ep.x - 46, ep.y - 12, 92, 22), Color("7c2d12"))
			o.draw_string(font, Vector2(ep.x - 42, ep.y + 4), "+%d queued" % backlog[k].size(), HORIZONTAL_ALIGNMENT_CENTER, 84, 12, Color.WHITE)
	# phase info in the junction centre
	var txt := ""
	if mode == "green":
		txt = "%d s" % int(green_t)
	elif mode == "yellow":
		txt = "Y"
	else:
		txt = "R"
	o.draw_circle(C, 15, Color(0, 0, 0, 0.55))
	o.draw_string(font, C + Vector2(-15, 5), txt, HORIZONTAL_ALIGNMENT_CENTER, 30, 13, Color.WHITE)
	# ambulance banner
	if reason_source == "emergency" or sim_t - ev_banner_t < 3.0:
		var on := int(Time.get_ticks_msec() / 350) % 2 == 0
		var r := Rect2(C.x - 260, 14, 520, 40)
		o.draw_rect(r, Color("b91c1c") if on else Color("7f1d1d"))
		o.draw_string(font, Vector2(r.position.x, r.position.y + 26), "AMBULANCE PRIORITY  -  %s road gets green" % NAMES[ev_banner_dir].to_upper(),
			HORIZONTAL_ALIGNMENT_CENTER, r.size.x, 15, Color.WHITE)
	if not running and sim_t == 0.0:
		var r2 := Rect2(C.x - 230, vp_h() - 64, 460, 40)
		o.draw_rect(r2, Color(0.04, 0.07, 0.12, 0.85))
		o.draw_string(font, Vector2(r2.position.x, r2.position.y + 26), "Set the traffic on the left, then press START",
			HORIZONTAL_ALIGNMENT_CENTER, r2.size.x, 15, Color("e5e7eb"))


func vp_h() -> float:
	return get_viewport_rect().size.y


# =================================================================== UI
func _style(bg: Color, radius := 8, border := Color(0, 0, 0, 0)) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = bg
	sb.set_corner_radius_all(radius)
	sb.set_content_margin_all(8)
	if border.a > 0:
		sb.border_color = border
		sb.set_border_width_all(1)
	return sb


func _make_theme() -> Theme:
	var th := Theme.new()
	th.default_font_size = 14
	th.set_stylebox("normal", "Button", _style(Color("1f2937"), 6, Color("374151")))
	th.set_stylebox("hover", "Button", _style(Color("273449"), 6, Color("60a5fa")))
	th.set_stylebox("pressed", "Button", _style(Color("1e3a5f"), 6, Color("60a5fa")))
	th.set_stylebox("disabled", "Button", _style(Color("161b22"), 6, Color("1f2937")))
	th.set_stylebox("focus", "Button", StyleBoxEmpty.new())
	for t in ["OptionButton"]:
		th.set_stylebox("normal", t, _style(Color("1f2937"), 6, Color("374151")))
		th.set_stylebox("hover", t, _style(Color("273449"), 6, Color("60a5fa")))
		th.set_stylebox("pressed", t, _style(Color("1e3a5f"), 6, Color("60a5fa")))
		th.set_stylebox("focus", t, StyleBoxEmpty.new())
	th.set_stylebox("normal", "LineEdit", _style(Color("0b1220"), 5, Color("374151")))
	th.set_stylebox("focus", "LineEdit", _style(Color("0b1220"), 5, Color("60a5fa")))
	th.set_color("font_color", "Label", Color("e5e7eb"))
	return th


func _label(text: String, size := 14, col := Color("e5e7eb"), wrap := false) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", col)
	if wrap:
		l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	return l


func _button(text: String, cb: Callable, min_w := 0) -> Button:
	var b := Button.new()
	b.text = text
	b.focus_mode = Control.FOCUS_NONE
	if min_w > 0:
		b.custom_minimum_size.x = min_w
	b.pressed.connect(cb)
	return b


func _spin(val: float, lo: float, hi: float, cb: Callable) -> SpinBox:
	var s := SpinBox.new()
	s.min_value = lo
	s.max_value = hi
	s.step = 1
	s.value = val
	s.custom_minimum_size.x = 74
	s.alignment = HORIZONTAL_ALIGNMENT_CENTER
	s.value_changed.connect(cb)
	return s


func _section(parent: Control, title: String) -> VBoxContainer:
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 5)
	parent.add_child(box)
	box.add_child(_label(title, 13, Color("93c5fd")))
	return box


func _panel(x: float, w: float) -> VBoxContainer:
	var p := PanelContainer.new()
	p.position = Vector2(x, 0)
	p.size = Vector2(w, 900)
	p.add_theme_stylebox_override("panel", _style(Color(0.04, 0.06, 0.1, 0.94), 0))
	ui["root"].add_child(p)
	var sc := ScrollContainer.new()
	sc.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	p.add_child(sc)
	var mc := MarginContainer.new()
	mc.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	for side in ["left", "right", "top", "bottom"]:
		mc.add_theme_constant_override("margin_" + side, 8)
	sc.add_child(mc)
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 10)
	v.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	mc.add_child(v)
	return v


func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var root := Control.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	root.theme = _make_theme()
	layer.add_child(root)
	ui["root"] = root

	# ------------------------------------------------ LEFT: controls
	var L := _panel(0, 330)
	var head := VBoxContainer.new()
	head.add_child(_label("SMART TRAFFIC JUNCTION", 20, Color.WHITE))
	head.add_child(_label("1 junction  |  4 signals  |  left-hand traffic", 12, Color("9ca3af")))
	L.add_child(head)

	var s1 := _section(L, "1  ALGORITHM")
	var ob := OptionButton.new()
	for i in ALGO_NAMES.size():
		ob.add_item(ALGO_NAMES[i], i)
	ob.selected = algo_idx
	ob.focus_mode = Control.FOCUS_NONE
	ob.item_selected.connect(_on_algo)
	s1.add_child(ob)
	ui["algo"] = ob
	var evc := CheckBox.new()
	evc.text = "Ambulance priority rule"
	evc.button_pressed = ev_priority
	evc.focus_mode = Control.FOCUS_NONE
	evc.toggled.connect(func(on): ev_priority = on)
	s1.add_child(evc)
	s1.add_child(_label("Switching algorithm replays the SAME traffic.", 11, Color("9ca3af"), true))

	var s2 := _section(L, "2  TRAFFIC ON EACH ROAD")
	var g := GridContainer.new()
	g.columns = 4
	g.add_theme_constant_override("h_separation", 6)
	g.add_theme_constant_override("v_separation", 6)
	for t in ["Road", "Waiting at\nstart *", "New cars\nper min", "Add now"]:
		g.add_child(_label(t, 11, Color("9ca3af")))
	for k in 4:
		var nm := _label(NAMES[k], 14, DIR_COLORS[k])
		nm.custom_minimum_size.x = 52
		g.add_child(nm)
		g.add_child(_spin(initial[k], 0, 40, func(v): initial[k] = int(v)))
		g.add_child(_spin(flows[k], 0, 30, func(v): _on_flow(k, v)))
		var hb := HBoxContainer.new()
		hb.add_child(_button("+1", func(): _add_random(k, 1)))
		hb.add_child(_button("+5", func(): _add_random(k, 5)))
		g.add_child(hb)
	s2.add_child(g)
	s2.add_child(_label("* used when you press Restart", 11, Color("9ca3af"), true))

	var s3 := _section(L, "3  ADD A SPECIAL VEHICLE")
	var hb2 := HBoxContainer.new()
	hb2.add_child(_label("to road:", 13, Color("cbd5e1")))
	var dob := OptionButton.new()
	for k in 4:
		dob.add_item(NAMES[k], k)
	dob.focus_mode = Control.FOCUS_NONE
	dob.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hb2.add_child(dob)
	s3.add_child(hb2)
	ui["dir"] = dob
	var tg := GridContainer.new()
	tg.columns = 3
	tg.add_theme_constant_override("h_separation", 6)
	tg.add_theme_constant_override("v_separation", 6)
	for i in TYPES.size():
		var t: String = TYPES[i]
		var b := _button(TYPE_LABELS[i], func(): add_vehicles(ui["dir"].selected, t, 1), 96)
		if t == "ambulance":
			b.add_theme_color_override("font_color", Color("fca5a5"))
		tg.add_child(b)
	s3.add_child(tg)
	var rac := CheckBox.new()
	rac.text = "Random ambulances (about 1 per minute)"
	rac.button_pressed = random_amb
	rac.focus_mode = Control.FOCUS_NONE
	rac.toggled.connect(func(on): random_amb = on)
	s3.add_child(rac)

	var s4 := _section(L, "4  RUN")
	var rb := HBoxContainer.new()
	var sb := _button("START", _on_start, 140)
	sb.add_theme_font_size_override("font_size", 16)
	ui["start"] = sb
	rb.add_child(sb)
	rb.add_child(_button("Restart", _on_restart, 80))
	rb.add_child(_button("New traffic", _on_new_traffic))
	s4.add_child(rb)
	var spb := HBoxContainer.new()
	spb.add_child(_label("Speed", 13, Color("cbd5e1")))
	for sp in [1, 2, 4, 8]:
		var b := _button("%dx" % sp, func(): _on_speed(sp), 50)
		ui["speed%d" % sp] = b
		spb.add_child(b)
	s4.add_child(spb)

	var s5 := _section(L, "5  RESULTS")
	var rr := HBoxContainer.new()
	var cb := _button("Compare algorithms", _on_compare)
	cb.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rr.add_child(cb)
	var db := _button("Dashboard", func(): OS.shell_open(BACKEND + "/"))
	db.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rr.add_child(db)
	s5.add_child(rr)
	var bl := _label("Backend: checking...", 12, Color("fbbf24"), true)
	ui["backend"] = bl
	s5.add_child(bl)

	# ------------------------------------------------ RIGHT: what is happening
	var R := _panel(1250, 350)
	var now := _section(R, "NOW")
	var gl := _label("North is GREEN", 24, Color("4ade80"))
	ui["green"] = gl
	now.add_child(gl)
	var rs := _label("", 13, Color("e5e7eb"), true)
	rs.custom_minimum_size = Vector2(320, 54)
	ui["reason"] = rs
	now.add_child(rs)

	var st := _section(R, "NUMBERS")
	var sg := GridContainer.new()
	sg.columns = 2
	sg.add_theme_constant_override("h_separation", 18)
	for key in ["Sim time", "Vehicles passed", "Avg wait (s)", "Ambulance wait (s)", "Stopped now", "Longest total queue"]:
		sg.add_child(_label(key, 13, Color("9ca3af")))
		var vl := _label("-", 15, Color.WHITE)
		ui["stat_" + key] = vl
		sg.add_child(vl)
	st.add_child(sg)

	var ch := _section(R, "VEHICLES STOPPED ON EACH ROAD")
	var chart := ChartScript.new()
	chart.custom_minimum_size = Vector2(320, 150)
	chart.colors = DIR_COLORS
	ui["chart"] = chart
	ch.add_child(chart)
	var lg := HBoxContainer.new()
	for k in 4:
		lg.add_child(_label("— " + NAMES[k] + "  ", 12, DIR_COLORS[k]))
	ch.add_child(lg)

	var how := _section(R, "HOW THIS ALGORITHM DECIDES")
	var ht := _label(ALGO_TEXT[algo_idx], 12, Color("cbd5e1"), true)
	ht.custom_minimum_size.x = 320
	ui["how"] = ht
	how.add_child(ht)
	how.add_child(_label("+ Ambulance rule (all algorithms): an ambulance within 120 m gets green after a safe yellow.", 12, Color("fca5a5"), true))

	var lgs := _section(R, "LAST SIGNAL CHANGES")
	var lgl := _label("", 11, Color("cbd5e1"), true)
	lgl.custom_minimum_size.x = 320
	ui["log"] = lgl
	lgs.add_child(lgl)

	# ------------------------------------------------ compare popup
	var pop := PanelContainer.new()
	pop.position = Vector2(225, 70)
	pop.size = Vector2(1150, 760)
	pop.add_theme_stylebox_override("panel", _style(Color("111827"), 12, Color("374151")))
	pop.visible = false
	root.add_child(pop)
	var pv := VBoxContainer.new()
	pop.add_child(pv)
	var ph := HBoxContainer.new()
	var pt := _label("Same traffic, every algorithm (5 simulated minutes, computed by the backend)", 15, Color.WHITE)
	pt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	ph.add_child(pt)
	ph.add_child(_button("Close", func(): pop.visible = false, 80))
	pv.add_child(ph)
	var pmsg := _label("", 14, Color("cbd5e1"), true)
	pv.add_child(pmsg)
	var tex := TextureRect.new()
	tex.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tex.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	tex.custom_minimum_size = Vector2(1130, 660)
	pv.add_child(tex)
	ui["pop"] = pop
	ui["pop_msg"] = pmsg
	ui["pop_tex"] = tex

	_on_speed(1)
	_set_online(false)
	ui["backend"].text = "Backend: checking..."


func _update_start_button() -> void:
	if ui.has("start"):
		ui["start"].text = "PAUSE" if running else ("START" if sim_t == 0.0 else "RESUME")


func _update_panel() -> void:
	if not ui.has("green"):
		return
	var nm: String = NAMES[current]
	if mode == "green":
		ui["green"].text = "%s is GREEN  (%d s)" % [nm, int(green_t)]
		ui["green"].add_theme_color_override("font_color", Color("4ade80"))
	elif mode == "yellow":
		ui["green"].text = "%s YELLOW -> %s" % [nm, NAMES[target]]
		ui["green"].add_theme_color_override("font_color", Color("fbbf24"))
	else:
		ui["green"].text = "ALL RED -> %s next" % NAMES[target]
		ui["green"].add_theme_color_override("font_color", Color("f87171"))
	var tag := {"emergency": "[AMBULANCE RULE] ", "min_green": "[MIN GREEN] ", "algorithm": "[%s] " % ALGO_NAMES[algo_idx].to_upper(), "info": ""}
	ui["reason"].text = tag.get(reason_source, "") + reason
	ui["reason"].add_theme_color_override("font_color", Color("fca5a5") if reason_source == "emergency" else Color("e5e7eb"))
	var stopped := 0
	if last_state.has("queue"):
		for x in last_state["queue"]:
			stopped += x
	ui["stat_Sim time"].text = "%d s" % int(sim_t)
	ui["stat_Vehicles passed"].text = str(passed)
	ui["stat_Avg wait (s)"].text = "%.1f" % (wait_sum / wait_n) if wait_n > 0 else "-"
	ui["stat_Ambulance wait (s)"].text = "%.1f  (%d)" % [amb_wait_sum / amb_n, amb_n] if amb_n > 0 else "-"
	ui["stat_Stopped now"].text = str(stopped)
	ui["stat_Longest total queue"].text = str(max_queue)
	ui["chart"].series = history
	ui["chart"].queue_redraw()
	var lines := []
	for d in decisions_log:
		lines.append("%4ds  %s" % [int(d["t"]), d["text"]])
	ui["log"].text = "\n".join(lines) if lines.size() > 0 else "No changes yet."
	_update_start_button()


# ------------------------------------------------------------------ UI callbacks
func _on_algo(i: int) -> void:
	var was := running
	_stop_run()
	algo_idx = i
	ui["how"].text = ALGO_TEXT[i]
	reset_sim()
	if was:
		_on_start()


func _on_flow(k: int, v: float) -> void:
	flows[k] = v
	next_arrival[k] = sim_t + _exp(v)


func _on_start() -> void:
	if running:
		running = false
		_stop_run()
	else:
		if not run_started:
			_start_run()
		running = true
	_update_start_button()


func _on_restart() -> void:
	_stop_run()
	reset_sim()


func _on_new_traffic() -> void:
	seed_val = randi() % 10000
	_on_restart()


func _on_speed(sp: int) -> void:
	speed = sp
	for x in [1, 2, 4, 8]:
		var b: Button = ui["speed%d" % x]
		b.add_theme_color_override("font_color", Color("60a5fa") if x == sp else Color("e5e7eb"))


func _on_compare() -> void:
	ui["pop"].visible = true
	if not online:
		ui["pop_msg"].text = "The backend is not running. Start it with:   python backend/server.py"
		ui["pop_tex"].texture = null
		return
	ui["pop_msg"].text = "Computing... (replays your current settings under every algorithm)"
	if http_compare.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		http_compare.cancel_request()
	http_compare.request(BACKEND + "/api/compare", ["Content-Type: application/json"], HTTPClient.METHOD_POST,
		JSON.stringify({"scenario": _scenario(), "ev_priority": ev_priority}))


func _on_compare_done(result: int, code: int, _h: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		ui["pop_msg"].text = "Comparison failed (code %d). Is the backend running?" % code
		return
	var img := Image.new()
	if img.load_png_from_buffer(body) != OK:
		ui["pop_msg"].text = "Could not read the chart."
		return
	ui["pop_tex"].texture = ImageTexture.create_from_image(img)
	ui["pop_msg"].text = "Shorter bars are better. The star marks the best. Ambulance rule: %s." % ("ON" if ev_priority else "OFF")


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_SPACE:
		_on_start()
