import tensorflow as tf
import numpy as np

sess = tf.Session()

# WORKS
def test1():
    class myClass():
        def __init__(self):
            self.p = tf.Variable(1.1,name='p')
            self.q = tf.Variable(1.0,name='q')
            pred = tf.equal(self.p,self.q)
            self.var1 = tf.Variable(3.0,name='var1')
            self.var2 = tf.Variable(2.0,name='var2')
            self.conditional = tf.cond(pred,lambda:self.var1,lambda: self.var2)
            
        def firstFunction(self):
            return self.conditional

    my_class = myClass()
    sess.run(tf.global_variables_initializer())
    sess.run(my_class.firstFunction())


# DOESN'T WORK
def test2():  
    class myClass():
        def __init__(self):
            self.firstFunction()
            
        def firstFunction(self):
            self.p = tf.Variable(1.1,name='p')
            self.q = tf.Variable(1.0,name='q')
            pred = tf.equal(self.p,self.q)
            self.var1 = tf.Variable(3.0,name='var1')
            self.var2 = tf.Variable(2.0,name='var2')
            self.conditional = tf.cond(pred,lambda:self.var1,lambda: self.var2)
            return self.conditional

    my_class = myClass()
    sess.run(tf.global_variables_initializer())
    sess.run(my_class.firstFunction())  # <---------------------------------------------- FAILS HERE

# WORKS
def test3():
    p = tf.Variable(1.1,name='p')
    q = tf.Variable(1.0,name='q')
    pred = tf.equal(p,q)
    var1 = tf.Variable(3.0,name='var1')
    var2 = tf.Variable(2.0,name='var2')
    conditional = tf.cond(pred,lambda:var1,lambda: var2)
    def firstFunction():
        return conditional

    sess.run(tf.global_variables_initializer())
    sess.run(firstFunction())

# DOESN'T WORK
def test4():
    def firstFunction():
        p = tf.Variable(1.1,name='p')
        q = tf.Variable(1.0,name='q')
        pred = tf.equal(p,q)
        var1 = tf.Variable(3.0,name='var1')
        var2 = tf.Variable(2.0,name='var2')
        conditional = tf.cond(pred,lambda:var1,lambda: var2)
        return conditional

    sess.run(tf.global_variables_initializer())
    sess.run(firstFunction())  # <---------------------------------------------- FAILS HERE

# WORKS
def test5():
    def firstFunction():
        p = tf.Variable(1.1,name='p')
        q = tf.Variable(1.0,name='q')
        pred = tf.equal(p,q)
        var1 = tf.Variable(3.0,name='var1')
        var2 = tf.Variable(2.0,name='var2')
        conditional = tf.cond(pred,lambda:var1,lambda: var2)
        sess.run(tf.global_variables_initializer())
        return conditional

    sess.run(firstFunction())

# WORKS
def test6():
    def firstFunction():
        place = tf.placeholder(tf.float32,name='place')
        p = tf.Variable(1.1,name='p')
        q = tf.Variable(1.0,name='q')
        pred = tf.equal(p,q)
        var1 = tf.Variable(3.0,name='var1')
        var2 = tf.Variable(2.0,name='var2')
        conditional = tf.cond(pred,lambda:var1,lambda: var2)
        sess.run(tf.global_variables_initializer())
        return conditional

    sess.run(firstFunction())

# WORKS
def test7():
    p = tf.placeholder(tf.float32,name='p')
    q = tf.Variable(1.0,name='q')
    pred = tf.equal(p,q)
    var1 = tf.Variable(3.0,name='var1')
    var2 = tf.Variable(2.0,name='var2')
    conditional = tf.cond(pred,lambda:var1,lambda: var2)
    sess.run(tf.global_variables_initializer())
    sess.run(conditional,feed_dict={p:float(1.0)})

# WORKS
def test8():
    p = tf.placeholder(tf.float32,name='p')
    def firstFunction():
        q = tf.Variable(1.0,name='q')
        pred = tf.equal(p,q)
        var1 = tf.Variable(3.0,name='var1')
        var2 = tf.Variable(2.0,name='var2')
        conditional = tf.cond(pred,lambda:var1,lambda: var2)
        sess.run(tf.global_variables_initializer())
        return conditional
    sess.run(firstFunction(),feed_dict={p:float(1.0)})

def test9():
    p = tf.placeholder(tf.float32,shape=(),name='p')
    q = tf.Variable(p,name='q')
    pred = tf.equal(p,q)
    var1 = tf.Variable(3.0,name='var1')
    var2 = tf.Variable(2.0,name='var2')
    conditional = tf.cond(pred,lambda:var1,lambda: var2)
    sess.run(tf.global_variables_initializer(),feed_dict={p:float(1.0)})
    # sess.run(firstFunction(),feed_dict={p:float(1.0)})



test9()
    