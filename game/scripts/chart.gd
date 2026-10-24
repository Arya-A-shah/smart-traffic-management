extends Control
## Small live line chart: vehicles stopped on each road, last N seconds.

var series := [[], [], [], []]
var colors := []
var window := 180


func _draw() -> void:
	var r := Rect2(Vector2.ZERO, size)
	draw_rect(r, Color("0b1220"))
	var top := 1.0
	for s in series:
		for x in s:
			top = maxf(top, float(x))
	top = maxf(5.0, ceilf(top / 5.0) * 5.0)
	var font := ThemeDB.fallback_font
	for i in 6:
		var y := size.y - 14 - (size.y - 22) * i / 5.0
		draw_line(Vector2(26, y), Vector2(size.x - 4, y), Color(1, 1, 1, 0.07), 1.0)
		draw_string(font, Vector2(2, y + 4), str(int(top * i / 5.0)), HORIZONTAL_ALIGNMENT_LEFT, 24, 10, Color("6b7280"))
	draw_string(font, Vector2(28, size.y - 1), "-%ds" % window, HORIZONTAL_ALIGNMENT_LEFT, 60, 10, Color("6b7280"))
	draw_string(font, Vector2(size.x - 34, size.y - 1), "now", HORIZONTAL_ALIGNMENT_LEFT, 40, 10, Color("6b7280"))
	for k in 4:
		var s: Array = series[k]
		if s.size() < 2:
			continue
		var pts := PackedVector2Array()
		var n := s.size()
		for i in n:
			var x := 26 + (size.x - 30) * (float(window - (n - 1 - i)) / window)
			var y := size.y - 14 - (size.y - 22) * float(s[i]) / top
			pts.append(Vector2(x, y))
		draw_polyline(pts, colors[k], 2.0, true)
