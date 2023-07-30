import datetime
import tensorflow as tf

def save_model(model, path):
	model.save(path)

def load_model(path):
	return tf.keras.models.load_model(path)

def autosave(model):
	dt = datetime.datetime.now()
	timestamp = dt.strftime("%Y-%m-%d--%H-%M-%S")
	save_model(model, "./backup/" + timestamp)