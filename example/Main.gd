extends Control

func _ready():
	var tokenizer = Tokenizer.new()
	add_child(tokenizer)
	
	print("--- Tokenizer Godot Example ---")
	
	# Cambia esta ruta al archivo .bin generado por el CLI de Rust
	var model_path = "../tokenizer.bin" 
	
	if tokenizer.load_model(model_path):
		var test_text = "I am loved by many of my followers"
		print("Original: ", test_text)
		
		var encoded = tokenizer.encode(test_text)
		print("Encoded (IDs): ", encoded)
		
		var decoded = tokenizer.decode(encoded)
		print("Decoded: ", decoded)
	else:
		print("Error: No se pudo cargar el modelo en: ", ProjectSettings.globalize_path(model_path))
		print("Asegúrate de haber ejecutado el programa Rust primero para generar tokenizer.bin")
