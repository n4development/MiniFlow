import tensorflow as tf

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
tf.matmul()

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={softmax: softmax})

    return output