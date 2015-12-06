##########################
# NEURAL NETWORK CLASSES #
##########################

import numpy as np
import theano
import theano.tensor as T
import timeit

class HiddenLayer(object):
    """
    Implementation of hidden layer class. Source: http://deeplearning.net/tutorial/mlp.html#mlp.

    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type input: theano.tensor.dmatrix
    :param input: a symbolic tensor of shape (n_examples, n_in)

    :type n_in: int
    :param n_in: dimensionality of input

    :type n_out: int
    :param n_out: number of hidden units

    :type activation: theano.Op or function
    :param activation: Non linearity to be applied in the hidden layer

    """

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        
        if W is None:
            if activation == theano.tensor.nnet.relu:
                W_values = np.asarray(
                    np.random.randn(n_in,n_out) * np.sqrt(2.0/n_in)
                    )
            else: 
                W_values = np.asarray(
                    rng.uniform(
                        low = -np.sqrt(6. / (n_in + n_out)), 
                        high = np.sqrt(6. / (n_in + n_out)), 
                        size = (n_in, n_out)
                    )
                    )
            
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        self.activation = activation
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]
        
class SimpleNN(object):
    ''' 
    Simple Neural Net class
    
    Single hidden layer net with one layer of hidden units and nonlinear activations. Top layer is linear
    Source: http://deeplearning.net/tutorial/mlp.html#mlp
    '''
    
    def __init__(self, rng, input, n_in, n_layers, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input (one minibatch)

        :type n_in: int
        :param n_in: number of input units
        
        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units

        """
        
        self.hiddenLayers = []
        
        for i in xrange(n_layers):
            if i == 0:
                layer_n_in = n_in
                layer_input = input
            else:
                layer_n_in = n_hidden
                layer_input = self.hiddenLayers[i-1].output
                
            layer_n_out = n_hidden
            
            hiddenLayer = HiddenLayer(
                rng = rng,
                input=layer_input,
                n_in = layer_n_in, 
                n_out = layer_n_out, 
                activation = T.nnet.relu
            )
            
            self.hiddenLayers.append(hiddenLayer)
        
        self.outputLayer = HiddenLayer(
            rng = rng,
            input = self.hiddenLayers[-1].output,
            n_in = n_hidden,
            n_out = n_out,
            activation = None
        )
             
        
        ## L1 and L2 regularization
        abs_W = np.array([])
        for i in xrange(n_layers):
            abs_W = np.append(abs_W, np.abs(self.hiddenLayers[i].W))
        abs_W = np.append(abs_W, np.abs(self.outputLayer.W))
        self.L1 = np.sum(abs_W)
        
        sq_W = 0.
        for i in xrange(n_layers):
            sq_W += T.sum(self.hiddenLayers[i].W**2)
        sq_W += T.sum(self.outputLayer.W**2)
        self.L2_sqr = sq_W
        
        self.params = []
        for i in xrange(n_layers):
            self.params = self.params + self.hiddenLayers[i].params
        self.params = self.params + self.outputLayer.params

        # Historical gradient for adagrad
        self.hist_grad = [theano.shared(value=np.zeros_like(param.get_value())) for param in self.params]
        
        self.input = input
        
        self.output = self.outputLayer.output
        
        
    def meanSqErr(self, u):
        if u.ndim != self.output.ndim:
            raise TypeError(
                'u should have the same shape as self.output',
                ('u', u.type, 'self.output', self.output.type)
            )
        return T.mean( (self.output - u)**2 )
    
    def meanAbsErr(self, u):
        if u.ndim != self.output.ndim:
            raise TypeError(
                'u should have the same shape as self.output',
                ('u', u.type, 'self.output', self.output.type)
            )
        return T.mean( T.abs_(self.output - u) )


###########################
# NEURAL NETWORK TRAINING #
###########################

def train_NN(ts_x, ts_u, vs_x, vs_u, te_x, te_u, # Train, validation, test data
             n_layers=1, n_hidden=20, # Network parameters
             learning_rule="Adagrad", learning_rate=1.0, L1_reg=0.0, L2_reg=0.0, batch_size=50, rho=0.9, #SGD parameters
             n_epochs=25, print_interval=1, random_order=True # Training parameters
            ):
    
    ################
    # PREPARE DATA #
    ################
    
    # Randomizes the order of the data
    def randomize_data(x_set, u_set):
        u_shape = u_set.shape
        all_data = np.c_[x_set, u_set]
        np.random.shuffle(all_data)
        x_new = all_data[:,:-u_shape[1]]
        u_new = all_data[:,-u_shape[1]:].reshape(u_shape)
        return x_new, u_new
    
    if random_order:
        ts_x, ts_u = randomize_data(ts_x, ts_u)
    
    x_dim = ts_x.shape[1]
    u_dim = ts_u.shape[1]
    
    train_set_x = theano.shared(
        value = ts_x, 
        name='ts_x', borrow=True)
    train_set_u = theano.shared(
        value = ts_u, 
        name='ts_u', borrow=True)
    valid_set_x = theano.shared(
        vs_x, 
        name='vs_x', borrow=True)
    valid_set_u = theano.shared(
        vs_u, 
        name='vs_u', borrow=True)
    test_set_x = theano.shared(
        te_x, 
        name='te_x', borrow=True)
    test_set_u = theano.shared(
        te_u, 
        name='te_u', borrow=True)
  
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as a matrix of positions
    u = T.matrix('u')  # control inputs are a 1d matrix.

    rng = np.random.RandomState(1234)

    n_in = x_dim
    n_out = u_dim
    
    # construct the MLP class
    
    classifier = SimpleNN(
        rng=rng,
        input=x,
        n_in=n_in,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_out=n_out
    )
    
    # define cost function
    ## NEED TO FIX
    cost = (
        classifier.meanSqErr(u)
        #+ L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    #return cost
    
    # compile a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            u: test_set_u[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            u: valid_set_u[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    # calculate gradient
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    # Fudge factor for adagrad, adadelta
    fudge_factor = 1E-5
    
    # rule for parameter updates: standard SGD or SGD with manual gradient decrease
    if (learning_rule == None) or (type(learning_rule) == tuple):
        updates = [
            (param, param- learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
        ]
    # Normalized SGD: divide grad by norm(grad)
    elif learning_rule == "normalized":
        updates = [
            (param, param-learning_rate * gparam / gparam.norm(L=2))
            for param, gparam in zip(classifier.params, gparams)
        ]
    
    # Adagrad
    elif learning_rule == "adagrad":
        adjusted_grad = [
            (gparam / (fudge_factor + T.sqrt(hist_g)))
             for gparam, hist_g in zip(gparams, classifier.hist_grad)
        ]
        updates = [
                (param, param-learning_rate * g_adj)
                for param, g_adj in zip(classifier.params, adjusted_grad)
            ]
    
        # update historical gradient - do before train model in adagrad
        hist_updates = [
                (hist_g, hist_g + gparam**2)
                for hist_g, gparam in zip(classifier.hist_grad, gparams)
            ]
    
    # Adadelta
    elif learning_rule == "adadelta":
        dx_accum = [theano.shared(value=np.zeros_like(param.get_value())) for param in classifier.params]
        
        # Calculate the updated adadelta gradient step dx
        adjusted_grad = [ 
            ( (T.sqrt(dx_acc) * gparam +fudge_factor)/ (T.sqrt(hist_g) + fudge_factor) )
            for gparam, hist_g, dx_acc in zip(gparams, classifier.hist_grad, dx_accum)
        ]
        
        # Update our gradient history
        hist_updates = [
            (hist_g, rho*hist_g + (1-rho)*gparam**2)
            for gparam, hist_g in zip(gparams, classifier.hist_grad)
        ]
        
        # Update our dx history
        dx_accum_updates = [
            (dx_acc, rho*dx_acc + (1-rho)*g_adj**2)
            for dx_acc, g_adj in zip(dx_accum, adjusted_grad)
        ]
        
        # Gradient step
        updates = [
            (param, param - learning_rate*g_adj)
            for param, g_adj, gparam in zip(classifier.params, adjusted_grad, gparams)
        ]
    
    # Compile our gradient history update if we are using adagrad or adadelta
    if (learning_rule == "adagrad") or (learning_rule == "adadelta"):
        update_hist = theano.function(
            inputs=[index],
            outputs=cost,
            updates=hist_updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                u: train_set_u[index * batch_size: (index + 1) * batch_size]
            }
        )
    
    # Compile our dx history update if we are using adadelta
    if (learning_rule == "adadelta"):
        update_dx_accum = theano.function(
            inputs=[index],
            outputs=cost,
            updates=dx_accum_updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                u: train_set_u[index * batch_size: (index + 1) * batch_size]
            }
        )
    
    # return cost and update parameters
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            u: train_set_u[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    
    # record training, validation, and test costs to be returned to user
    training_losses = np.array([]) ##
    validation_losses = np.array([]) ##
    test_losses = np.array([]) ##

    # early-stopping parameters: NOTE - AREN'T REALLY BEING USED NOW
    patience = 100000000  # look as this many examples regardless (10000)
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    test_score = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    avg_training_loss = 0.
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            # update our sum of training losses for this minibatch
            if (learning_rule == "adagrad") or (learning_rule == "adadelta"):
                update_hist(minibatch_index)
            minibatch_avg_cost = train_model(minibatch_index)
            
            if (learning_rule == "adadelta"):
                update_dx_accum(minibatch_index)
            
            avg_training_loss += minibatch_avg_cost
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # update our list of training losses
                training_losses = np.append(training_losses, avg_training_loss / validation_frequency)
                avg_training_loss = 0
                
                # apply our rule to update the learning rate
                if type(learning_rule) == tuple:
                    if (iter / validation_frequency == learning_rule[0]):
                        learning_rate = learning_rule[1]*learning_rate
                
                # compute validation error
                these_validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(these_validation_losses)
                
                # update our list of validation losses
                validation_losses = np.append(validation_losses, this_validation_loss)
                
              

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    epoch_test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(epoch_test_losses)
                    
                    # update our list of test scores
                    test_losses = np.append(test_losses, test_score)
                        
                # print progress if we are at the print interval
                if (iter + 1) % (validation_frequency * print_interval) == 0:
                    print(
                        ('epoch %i, minibatch %i/%i, validation error %f, test error of best model %f') %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss,
                            test_score
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f '
           'obtained at iteration %i, with test performance %f') %
          (best_validation_loss, best_iter + 1, test_score))
    print 'The code ran for %d epochs, with %f epochs/sec' % (
        epoch, 1.*epoch / (end_time - start_time))
    
    return x, classifier, training_losses, validation_losses, test_losses
    
    
