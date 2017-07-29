import tensorflow as tf

'''
Structure of the neural network
[input_data]->(weights)->[hidden_layer_1]->(activation_function)->(weights)->[hidden_layer_2]->(activation_function)->(weights)->[output layer}        [FeedFoward Neural Network]

compare output to the desired output [cost function]

(optimize the network using the cost function.....Some optimizers that could be used...SGD,AdaGrad)

[backpropagation]:The process of updating the weights after the optimization

feedforward+backpropgation=one cycle

repeat the cycle until the error is very low.

usually it will take us about 10 cycles.Let's see how it goes.
'''

#importing the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

#10 classes , 0-9
'''
How the one_hot method will work
only the actual output is marked as '1' rest all being '0'
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
2=[0,0,1,0,0,0,0,0,0,0]
.
.
'
'''

#inintializing some required variables 
n_nodes_h1=500
n_nodes_h2=500
n_nodes_h3=500

n_classes=10 #0-9 numbers
batch_size=100#The whole input data is divided into batches of 100.

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')


def neural_net_model(data):
    hidden_1_layer={
            'weights':tf.Variable(tf.random_normal([784,n_nodes_h1])),
             'biases':tf.Variable(tf.random_normal([n_nodes_h1]))
                    }
     
    hidden_2_layer={
            'weights':tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])),
            'biases':tf.Variable(tf.random_normal([n_nodes_h2]))
                    }
    
    hidden_3_layer={
            'weights':tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3])),
             'biases':tf.Variable(tf.random_normal([n_nodes_h3]))
                   }
    
    output_layer={
            'weights':tf.Variable(tf.random_normal([n_nodes_h3,n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))
                 }
    
    #Our model's structure->(input_data*weights)+biases
    #Here we have used the RELU (Rectified Linear Unit) function as our activation function, rather than using sigmoid. RELU is fast and handles the gradient error well.
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)
     
    output=tf.matmul(l3,output_layer['weights'])+output_layer['biases']
                 
    return output

def train_neural_network(x):
    prediction=neural_net_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    
    #learning_rate = 0.001 by default
    optimizer=tf.train.AdamOptimizer().minimize(cost) 
    
    '''
        We have used the AdamOptimizer here, we could also have gone for the GradientDescentOptimizer.
        The reason behind choosing the AdamOptimizer is its average moving parameters which helps in reaching the convergence stage faster, by taking larger effective step size.
        
    '''
    
    #number of cycles=10
    #one cycle=feedforward+backpropagation
    n_epochs=10
    
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss=0
            for _ in range(int (mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                _,c=session.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss+=c
            print 'Epoch '+str(epoch+1) +' completed out of '+str(n_epochs)+' Loss: '+str(epoch_loss)
    
    
        is_correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(is_correct,'float')) #Here the cast funtion is used to change the datatype of the is_correct variable.
        print 'Accuracy: '+str(accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
    

train_neural_network(x)
