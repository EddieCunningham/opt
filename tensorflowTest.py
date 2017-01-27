import tensorflow as tf

def firstExample():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x,W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def tensorflowExample():
    from tensorflow.examples.tutorials.mnist import input_data

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])

    # --------------------------------------------------------

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # --------------------------------------------------------

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # --------------------------------------------------------

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # --------------------------------------------------------

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # --------------------------------------------------------

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # --------------------------------------------------------

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(200):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def notLoopTest():

    # Define a single queue with two components to store the input data.                                                                                           
    q_data = tf.FIFOQueue(100000, [tf.float32, tf.float32])

    # We will use these placeholders to enqueue input data.                                                                                                        
    placeholder_x = tf.placeholder(tf.float32, shape=[None])
    placeholder_y = tf.placeholder(tf.float32, shape=[None])
    enqueue_data_op = q_data.enqueue_many([placeholder_x, placeholder_y])

    gs = tf.Variable(0)
    w = tf.Variable(0.)
    b = tf.Variable(0.)

    optimizer = tf.train.GradientDescentOptimizer(0.05)

    # Construct the while loop.                                                                                                                                    
    def cond(i):
        return i < 10

    def body(i):
        # Dequeue a single new example each iteration.                                                                                                                      
        x, y = q_data.dequeue()

        # Compute the loss and gradient update based on the current example.                                                                                         
        loss = (tf.add(tf.multiply(x, w), b) - y)**2
        train_op = optimizer.minimize(loss, global_step=gs)

        # Ensure that the update is applied before continuing.                                                                                                       
        with tf.control_dependencies([train_op]):
            return i + 1

    loop = tf.while_loop(cond, body, [tf.constant(0)])

    data = [k*1. for k in range(10)]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for _ in range(10):
            # NOTE: Constructing the enqueue op ahead of time avoids adding                                                                                            
            # (potentially many) copies of `data` to the graph.                                                                                                        
            sess.run(body, feed_dict={placeholder_x: data,
                                                 placeholder_y: data})

        print sess.run([gs, w, b])  # Prints before-loop values.                                                                                                     
        sess.run(loop)
        print sess.run([gs, w, b])  # Prints after-loop values.



def loopTest():

    # Define a single queue with two components to store the input data.                                                                                           
    q_data = tf.FIFOQueue(100000, [tf.float32, tf.float32])

    # We will use these placeholders to enqueue input data.                                                                                                        
    placeholder_x = tf.placeholder(tf.float32, shape=[None])
    placeholder_y = tf.placeholder(tf.float32, shape=[None])
    enqueue_data_op = q_data.enqueue_many([placeholder_x, placeholder_y])

    gs = tf.Variable(0)
    w = tf.Variable(0.)
    b = tf.Variable(0.)

    optimizer = tf.train.AdamOptimizer(0.05)

    # Construct the while loop.                                                                                                                                    
    def cond(i):
        return i < 10000

    def body(i):
        # Dequeue a single new example each iteration.                                                                                                                      
        x, y = q_data.dequeue()

        # Compute the loss and gradient update based on the current example.                                                                                         
        loss = (tf.add(tf.multiply(x, w), b) - y)**2
        train_op = optimizer.minimize(loss)

        # Ensure that the update is applied before continuing.                                                                                                       
        with tf.control_dependencies([train_op]):
            return i + 1

    loop = tf.while_loop(cond, body, [tf.constant(0)])

    data = [k*1. for k in range(10000)]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for _ in range(1):
            # NOTE: Constructing the enqueue op ahead of time avoids adding                                                                                            
            # (potentially many) copies of `data` to the graph.                                                                                                        
            sess.run(enqueue_data_op, feed_dict={placeholder_x: data,
                                                 placeholder_y: data})

        print sess.run([gs, w, b])  # Prints before-loop values.                                                                                                     
        sess.run(loop)
        print sess.run([gs, w, b])  # Prints after-loop values.



loopTest()









