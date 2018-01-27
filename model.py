import tensorflow as tf
from PIL import Image
import numpy as np

sess = tf.Session()
saver = tf.train.import_meta_graph("model.14.ckpt.meta")
saver.restore(sess, "model.14.ckpt")
graph = tf.get_default_graph()

input_placeholder = graph.get_tensor_by_name("Placeholder:0")

image = Image.open("sports_images/liverpool-chelsea.jpeg")
image = image.resize([640, 480], Image.ANTIALIAS)

clas_examples = graph.get_tensor_by_name("Sum:0")
clas_answer = graph.get_tensor_by_name("ArgMax:0")
clas_guess = graph.get_tensor_by_name("ArgMax_1:0")
clas_prob = graph.get_tensor_by_name("Softmax:0")
reg_rpn = graph.get_tensor_by_name("Reshape_1:0")

clas_placeholder = graph.get_tensor_by_name("Placeholder_1:0")
reg_placeholder = graph.get_tensor_by_name("Placeholder_2:0")

train_step = graph.get_tensor_by_name("Momentum:0")

output = sess.run([clas_placeholder, reg_placeholder], feed_dict={ input_placeholder: [np.array(image)]})
