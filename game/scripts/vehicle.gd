extends Node2D
## One vehicle. It drives along a Curve2D (approach -> junction -> exit road).
## `s` is the distance of its FRONT bumper along that path, in pixels.

const PX := 6.0   # pixels per metre

const SPECS := {
	# length/width in metres, top speed in m/s, acceleration in m/s^2
	"car":       {"len": 4.6, "wid": 2.1, "vmax": 12.5, "acc": 2.6},
	"bike":      {"len": 2.2, "wid": 0.9, "vmax": 13.0, "acc": 3.6},
	"auto":      {"len": 3.2, "wid": 1.5, "vmax": 10.0, "acc": 2.2},
	"bus":       {"len": 10.5, "wid": 2.6, "vmax": 11.0, "acc": 1.3},
	"truck":     {"len": 8.5, "wid": 2.5, "vmax": 10.5, "acc": 1.2},
	"ambulance": {"len": 5.4, "wid": 2.3, "vmax": 13.0, "acc": 2.8},
}
const CAR_COLORS := ["e5e7eb", "94a3b8", "1e293b", "dc2626", "2563eb", "0f766e", "ca8a04", "7c3aed", "f8fafc"]

var id := 0
var dir := 0            # approach it comes from: 0 N, 1 E, 2 S, 3 W
var turn := 0           # 0 straight, 1 left, 2 right
var vtype := "car"
var length := 28.0
var width := 13.0
var vmax := 75.0
var accel := 15.0
var body_color := Color.WHITE
var curve: Curve2D
var path_len := 0.0
var stop_s := 0.0       # where the stop line is on the path
var jstart_s := 0.0     # entering the junction box
var jend_s := 0.0       # leaving the junction box
var exit_idx := 0       # which road it leaves on
var s := 0.0
var v := 0.0
var spawn_t := 0.0
var wait := 0.0         # seconds spent (almost) stopped
var passed_line := false
var braking := false
var t_anim := 0.0


func setup(p_type: String, rng: RandomNumberGenerator) -> void:
	vtype = p_type
	var sp: Dictionary = SPECS[vtype]
	length = sp["len"] * PX
	width = sp["wid"] * PX
	vmax = sp["vmax"] * PX * rng.randf_range(0.92, 1.05)
	accel = sp["acc"] * PX
	match vtype:
		"car":
			body_color = Color(CAR_COLORS[rng.randi_range(0, CAR_COLORS.size() - 1)])
		"bike":
			body_color = Color(["111827", "1d4ed8", "b91c1c", "374151"][rng.randi_range(0, 3)])
		"auto":
			body_color = Color("16a34a")
		"bus":
			body_color = Color(["b91c1c", "1d4ed8", "c2410c"][rng.randi_range(0, 2)])
		"truck":
			body_color = Color("1e40af")
		"ambulance":
			body_color = Color("f8fafc")


func place() -> void:
	var tr := curve.sample_baked_with_rotation(clampf(s, 0.0, path_len - 0.01))
	position = tr.origin
	rotation = tr.get_rotation()


func _process(delta: float) -> void:
	t_anim += delta
	if vtype == "ambulance":
		queue_redraw()


func _draw() -> void:
	# local frame: +x = forward, front bumper at x = 0, body spans x in [-length, 0]
	var L := length
	var W := width
	var hw := W / 2.0
	draw_rect(Rect2(-L + 2, -hw + 2, L, W), Color(0, 0, 0, 0.35))   # shadow
	match vtype:
		"bike":
			draw_rect(Rect2(-L, -hw * 0.6, L, W * 0.6), Color("0f172a"))
			draw_rect(Rect2(-L * 0.8, -hw, L * 0.55, W), body_color)
			draw_circle(Vector2(-L * 0.5, 0), hw * 1.05, Color("fbbf24"))     # helmet
		"auto":
			draw_rect(Rect2(-L, -hw, L, W), Color("facc15"))
			draw_rect(Rect2(-L * 0.92, -hw + 1, L * 0.7, W - 2), Color("111827"))  # canopy
			draw_rect(Rect2(-L * 0.22, -hw * 0.55, L * 0.2, W * 0.55), body_color)
		"bus":
			draw_rect(Rect2(-L, -hw, L, W), body_color)
			draw_rect(Rect2(-L + 3, -hw + 2.5, L - 9, W - 5), body_color.lightened(0.18))
			var n := 7
			for i in n:
				var x := -L + 5 + i * (L - 12) / n
				draw_rect(Rect2(x, -hw, (L - 12) / n - 2, 2.2), Color("bae6fd"))
				draw_rect(Rect2(x, hw - 2.2, (L - 12) / n - 2, 2.2), Color("bae6fd"))
			draw_rect(Rect2(-5, -hw + 2, 3.5, W - 4), Color("bae6fd"))
		"truck":
			draw_rect(Rect2(-L, -hw, L * 0.74, W), Color("f59e0b"))           # cargo
			draw_line(Vector2(-L + 3, 0), Vector2(-L * 0.3, 0), Color("b45309"), 1.0)
			draw_rect(Rect2(-L * 0.24, -hw + 0.5, L * 0.24, W - 1), body_color)  # cab
			draw_rect(Rect2(-5, -hw + 2, 3, W - 4), Color("bae6fd"))
		"ambulance":
			draw_rect(Rect2(-L, -hw, L, W), body_color)
			draw_rect(Rect2(-L, -1.6, L, 3.2), Color("dc2626"))
			var cx := -L * 0.55
			draw_rect(Rect2(cx - 4.5, -1.5, 9, 3), Color("dc2626"))
			draw_rect(Rect2(cx - 1.5, -4.5, 3, 9), Color("dc2626"))
			draw_rect(Rect2(-6, -hw + 2, 3.5, W - 4), Color("1e293b"))
			var on := int(t_anim * 6.0) % 2 == 0
			draw_rect(Rect2(-L * 0.28, -hw, 3.5, hw), Color("ef4444") if on else Color("7f1d1d"))
			draw_rect(Rect2(-L * 0.28, 0, 3.5, hw), Color("1e3a8a") if on else Color("3b82f6"))
			var glow := Color(1, 0.2, 0.2, 0.22) if on else Color(0.2, 0.4, 1, 0.22)
			draw_circle(Vector2(-L * 0.5, 0), L * 0.85, glow)
		_:  # car
			draw_rect(Rect2(-L, -hw, L, W), body_color)
			draw_rect(Rect2(-L * 0.36, -hw + 1.5, L * 0.15, W - 3), Color("0f172a"))   # windscreen
			draw_rect(Rect2(-L * 0.86, -hw + 1.5, L * 0.1, W - 3), Color("1e293b"))    # rear glass
			draw_rect(Rect2(-L * 0.7, -hw + 2, L * 0.33, W - 4), body_color.darkened(0.08))
	# head lights / brake lights
	if vtype != "bike":
		draw_rect(Rect2(-1.5, -hw + 0.5, 1.5, 2.2), Color("fef9c3"))
		draw_rect(Rect2(-1.5, hw - 2.7, 1.5, 2.2), Color("fef9c3"))
	if braking or v < 2.0:
		draw_rect(Rect2(-L, -hw + 0.5, 1.8, 2.4), Color("ff2d2d"))
		draw_rect(Rect2(-L, hw - 2.9, 1.8, 2.4), Color("ff2d2d"))
