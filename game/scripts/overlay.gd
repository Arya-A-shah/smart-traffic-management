extends Node2D
## Drawn above the vehicles: signal heads, labels, banners. Main does the work.

var main


func _draw() -> void:
	if main:
		main.draw_overlay(self)
