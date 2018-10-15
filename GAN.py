import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import midi


tf.reset_default_graph()

def app_run(_epochs,_batch_size,_d_lr,_g_lr):

    lower_bound = 24
    upper_bound = 102
    span = upper_bound - lower_bound


    def midiToNoteStateMatrix(midi_file_path, squash=True, span=span):
        pattern = midi.read_midifile(midi_file_path)
     
        time_left = []
        for track in pattern:
            time_left.append(track[0].tick)
        
        posns = [0 for track in pattern]
     
        statematrix = []
        time = 0
     
        state = [[0,0] for x in range(span)]
        statematrix.append(state)
        condition = True
        while condition:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(span)]
                statematrix.append(state)
            for i in range(len(time_left)):
                if not condition:
                    break
                while time_left[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
     
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
                            pass
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-lower_bound] = [0, 0]
                            else:
                                state[evt.pitch-lower_bound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            out =  statematrix
                            condition = False
                            break
                    try:
                        time_left[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        time_left[i] = None
     
                if time_left[i] is not None:
                    time_left[i] -= 1
     
            if all(t is None for t in time_left):
                break
     
            time += 1
     
        S = np.array(statematrix)
        statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
        statematrix = np.asarray(statematrix).tolist()
        return statematrix
     
    def noteStateMatrixToMidi(statematrix, filename="output_file", span=span):
        statematrix = np.array(statematrix)
        if not len(statematrix.shape) == 3:
            statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
        statematrix = np.asarray(statematrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        
        span = upper_bound-lower_bound
        tickscale = 55
        
        lastcmdtime = 0
        prevstate = [[0,0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):  
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lower_bound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lower_bound))
                lastcmdtime = time
                
            prevstate = state
        
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
     
        midi.write_midifile("{}.mid".format(filename), pattern)
     
    def get_songs(midi_path):
        files = os.listdir(midi_path)
        songs = []
        for f in files:
            f = midi_path+'/'+f
            print('loading:', f)
            try:
                song = np.array(midiToNoteStateMatrix(f))
                if np.array(song).shape[0] > 64:
                    songs.append(song)
            except Exception as e:
                print('shu ju wu xiao: ', e)
        print("the midi files number is: ", len(songs))
        return songs


    def device_for_node(n):
            if n.type == "MatMul":
                return "/gpu:0"
            else:
                return "/cpu:0"

    def add_layer(inputs, Weights, biases, activation_function=None, norm=False):
        # weights and biases (bad initialization for this case)
    #     Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
    #     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

        # fully connected product
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # normalize fully connected product
        if norm:
            # Batch Normalize
            fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones(tf.shape(biases)))
            shift = tf.Variable(tf.zeros(tf.shape(biases)))
            epsilon = 0.001

            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()

            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
            # similar with this two steps:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        # activation
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs


    def toMusic(sample):
        for i in range(0,len(sample)):
            for j in range(0,len(sample[i])):
                sample[i][j] = round(sample[i][j])
        return sample



    # inputs
    def get_inputs():
         
        inputs_real = tf.placeholder(tf.float32, [None, n_input], name='inputs_real')
        inputs_noise = tf.placeholder(tf.float32, [None, n_input], name='inputs_noise')
        
        return inputs_real, inputs_noise


    # Generator
    def get_generator(noise_img, is_train=True, alpha=0.01):

        with tf.variable_scope("generator", reuse=(not is_train)):

            h1 = add_layer(noise_img, w_layer1, b_layer1, activation_function=tf.nn.sigmoid, norm=None)
            h2 = add_layer(h1, w_layer2, b_layer2, activation_function=tf.nn.sigmoid, norm=None)
            h3 = add_layer(h2, w_layer3, b_layer3, activation_function=tf.nn.sigmoid, norm=None)
            h4 = add_layer(h3, w_layer4, b_layer4, activation_function=tf.nn.sigmoid, norm=None)
            h5 = add_layer(h4, w_layer5, b_layer5, activation_function=tf.nn.sigmoid, norm=None)
            pred = add_layer(h5, w_layer6, b_layer6, activation_function=tf.nn.sigmoid, norm=None)
            
            print(pred.shape)

            return pred


    # Discriminator
    def get_discriminator(inputs_img, reuse=False, alpha=0.01):
        
        with tf.variable_scope("discriminator", reuse=reuse):

            h1 = add_layer(inputs_img, D_w_layer1, D_b_layer1, activation_function=tf.nn.sigmoid, norm=None)
            h2 = add_layer(h1, D_w_layer2, D_b_layer2, activation_function=tf.nn.sigmoid, norm=None)
            h3 = add_layer(h2, D_w_layer3, D_b_layer3, activation_function=tf.nn.sigmoid, norm=None)
            logits = add_layer(h3, D_w_layer4, D_b_layer4, activation_function=None, norm=None)
            
            outputs = tf.sigmoid(logits)
            
            return logits, outputs

        
    # Loss
    def get_loss(inputs_real, inputs_noise, smooth=0.1):
       
        
        g_outputs = get_generator(inputs_noise, is_train=True)
        d_logits_real, d_outputs_real = get_discriminator(inputs_real)
        d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
        
        # get Loss
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                        labels=tf.ones_like(d_outputs_fake)*(1-smooth)))
        
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                             labels=tf.ones_like(d_outputs_real)*(1-smooth)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                             labels=tf.zeros_like(d_outputs_fake)))
        d_loss = tf.add(d_loss_real, d_loss_fake)
        
        return g_loss, d_loss


    # Optimizer
    def get_optimizer(g_loss, d_loss, beta1=0.4, d_learning_rate=0.0005, g_learning_rate=0.0002):
        
        
        train_vars = tf.trainable_variables()
        
        g_vars = [var for var in train_vars if var.name.startswith("generator")]
        d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
        
        # Optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            g_opt = tf.train.AdamOptimizer(g_learning_rate).minimize(g_loss)
            d_opt = tf.train.AdamOptimizer(d_learning_rate).minimize(d_loss)
        
        return g_opt, d_opt



    # def plot_images(samples,i):
    #     plt.figure()
    #     samples = (samples + 1) / 2
    #     fig, axes = plt.subplots(nrows=1, ncols=25, sharex=True, sharey=True, figsize=(50,2))
    #     for img, ax in zip(samples, axes):
    #         ax.imshow(img.reshape((96, 96, 3)), cmap='Greys_r')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #     fig.tight_layout(pad=0)
    #     plt.savefig("examples"+str(i)+".jpg")

        
    # def show_generator_output(sess, n_images, inputs_noise, output_dim):
    #     """
        
    #     """
    #     cmap = 'Greys_r'
    #     noise_shape = inputs_noise.get_shape().as_list()[-1]
        
    #     examples_noise = np.random.normal(-1, 1, size=[n_images, noise_shape])

    #     samples = sess.run(get_generator(inputs_noise, output_dim, False),
    #                        feed_dict={inputs_noise: examples_noise})


    #     return samples

    def train_Auto():

        def fun():
            creat_music = []
            flag = False
            songs = get_songs('midi_train')
            print(np.array(songs).shape)
            for epoch in range(epochs):
                loss_train = 0
                count_loss = 0
                print(epoch)
                for song in songs:
                    song = np.array(song)
                    # print(song.shape)     #(273,156)
                    song = song[:int(np.floor(song.shape[0]/n_timesteps) * n_timesteps)]
                    song = np.reshape(song, [song.shape[0]//n_timesteps, song.shape[1] * n_timesteps])
                    # print(song.shape)   #(2,19968)
                    # return
                    for i in range(0, len(song), batch_size): 
                        train_x = song[i:i+batch_size]
                        if flag == False and epoch == 4:
                            for k in range(0,len(train_x)):
                                creat_music.append(train_x[k])
                                print("coming!!!")
                            flag = True
                            print(creat_music)
                            print(np.array(creat_music).shape)
                        # print(train_x.shape)

                        sess.run(train_op, feed_dict={X: train_x,Y: train_x})
                        loss_train = loss_train + sess.run(cost, feed_dict={X: train_x,Y: train_x})
                        count_loss = count_loss + 1
                print(loss_train/count_loss)

                if epoch == epochs - 1:
                    saver.save(sess, 'midi.module')
                
            print(np.array(creat_music).shape)
            # sample = gibbs_sample(1).eval(session=sess, feed_dict={X: creat_music})
            sample = (sess.run(pred, feed_dict={X: creat_music,Y: creat_music}))
            sample = toMusic(sample)
            print(np.array(sample).shape)
            print("sample")
            print(sample)
            S = np.reshape(sample[0,:], (n_timesteps, 2 * note_range))
            print(S.shape)
            noteStateMatrixToMidi(S, "auto_gen_music_auto")
            print('creat auto_gen_music.mid  file')
            return


        h1 = add_layer(X, w_layer1, b_layer1, activation_function=tf.nn.sigmoid, norm=None)
        h2 = add_layer(h1, w_layer2, b_layer2, activation_function=tf.nn.sigmoid, norm=None)
        h3 = add_layer(h2, w_layer3, b_layer3, activation_function=tf.nn.sigmoid, norm=None)
        h4 = add_layer(h3, w_layer4, b_layer4, activation_function=tf.nn.sigmoid, norm=None)
        h5 = add_layer(h4, w_layer5, b_layer5, activation_function=tf.nn.sigmoid, norm=None)
        pred = add_layer(h5, w_layer6, b_layer6, activation_function=tf.nn.sigmoid, norm=None)


        with tf.name_scope('loss'):
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
            cost = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(Y, pred)), 0, True))

        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(0.005).minimize(cost)

        with tf.Session() as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.all_variables())
            sess.run(init)
          
            with sess.graph.as_default():
                with sess.graph.device(device_for_node):
                    fun()


    #Train
    def train_GAN():
        
        creat_music = []
        flag = False
        
        # losses = []
        # steps = 0
        # num = 1
        
        songs = get_songs('midi_train')

        inputs_real, inputs_noise = get_inputs()
        g_loss, d_loss = get_loss(inputs_real, inputs_noise)
        g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, d_learning_rate, g_learning_rate)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # print(sess.run(w_layer1))
            # print(sess.run(b_layer1))
            # print(ca)
            # for e in range(epochs):
            #     for batch_i in range(images.shape[0]//batch_size-1):
            #         steps += 1

            #         batch_images = images[batch_i * batch_size: (batch_i+1)*batch_size]
            #         # scale to -1, 1
            #         batch_images = batch_images * 2 - 1

            #         # noise
            #         batch_noise = np.random.normal(-1, 1, size=(batch_size, noise_size))

            #         # run optimizer
            #         _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
            #                                              inputs_noise: batch_noise})
            #         _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
            #                                              inputs_noise: batch_noise})
                    
            #         if steps % 101 == 0:
            #             train_loss_d = d_loss.eval({inputs_real: batch_images,
            #                                         inputs_noise: batch_noise})
            #             train_loss_g = g_loss.eval({inputs_real: batch_images,
            #                                         inputs_noise: batch_noise})
            #             losses.append((train_loss_d, train_loss_g))
            #             # show pic
            #             samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
            #             plot_images(samples,num)
            #             num = num + 1
            #             print("Epoch {}/{}....".format(e+1, epochs), 
            #                   "Discriminator Loss: {:.4f}....".format(train_loss_d),
            #                   "Generator Loss: {:.4f}....". format(train_loss_g))
            for epoch in range(epochs2):
                g_loss_train = 0
                d_loss_train = 0
                count_loss = 0
                print(epoch)
                for song in songs:
                    song = np.array(song)
                    # print(song.shape)     #(273,156)
                    song = song[:int(np.floor(song.shape[0]/n_timesteps) * n_timesteps)]
                    song = np.reshape(song, [song.shape[0]//n_timesteps, song.shape[1] * n_timesteps])
                    # print(song.shape)   #(2,19968)
                    # return
                    for i in range(0, len(song), batch_size): 
                        train_x = song[i:i+batch_size]
                        if flag == False and epoch == 4:
                            for k in range(0,len(train_x)):
                                creat_music.append(train_x[k])
                                print("coming!!!")
                            flag = True
                            print(creat_music)
                            print(np.array(creat_music).shape)
                        # print(train_x.shape)

                        # run optimizer
                        _ = sess.run(g_train_opt, feed_dict={inputs_real: train_x,
                                                         inputs_noise: train_x})
                        _ = sess.run(d_train_opt, feed_dict={inputs_real: train_x,
                                                          inputs_noise: train_x})

                        train_loss_g = g_loss.eval({inputs_real: train_x,
                                                    inputs_noise: train_x})
                        train_loss_d = d_loss.eval({inputs_real: train_x,
                                                    inputs_noise: train_x})
                        g_loss_train = g_loss_train + train_loss_g
                        d_loss_train = d_loss_train + train_loss_d
                        count_loss = count_loss + 1
                print('g_loss_train: ' + str(g_loss_train/count_loss))
                print('d_loss_train: ' + str(d_loss_train/count_loss))


            
            print(np.array(creat_music).shape)
            sample = sess.run(get_generator(inputs_noise, is_train=False),
                           feed_dict={inputs_noise: creat_music})
            sample = toMusic(sample)
            print(np.array(sample).shape)
            print("sample")
            print(sample)
            S = np.reshape(sample[0,:], (n_timesteps, 2 * note_range))
            print(S.shape)
            noteStateMatrixToMidi(S, "auto_gen_music_gan")
            print('creat auto_gen_music.mid  file')
            return             
                        
                        


    note_range = upper_bound - lower_bound
    n_timesteps = 128
    n_input = 2 * note_range * n_timesteps
    epochs = 256
    epochs2 = _epochs
    batch_size = _batch_size
    d_learning_rate = _d_lr
    g_learning_rate = _g_lr
    # d_learning_rate = 0.0003
    # g_learning_rate = 0.0002
    beta1 = 0.4
    n_samples = 25


    n_hidden_layer1 = 128
    n_hidden_layer2 = 64
    n_hidden_layer3 = 32
    n_hidden_layer4 = 64
    n_hidden_layer5 = 128
    n_output = n_input


    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_output])

    with open('w_layer1.pik', 'rb') as f:  
        w1 = pickle.load(f)
    with open('w_layer2.pik', 'rb') as f:  
        w2 = pickle.load(f)
    with open('w_layer3.pik', 'rb') as f:  
        w3 = pickle.load(f)
    with open('w_layer4.pik', 'rb') as f:  
        w4 = pickle.load(f)
    with open('w_layer5.pik', 'rb') as f:  
        w5 = pickle.load(f)
    with open('w_layer6.pik', 'rb') as f:  
        w6 = pickle.load(f)

    with open('b_layer1.pik', 'rb') as f:  
        b1 = pickle.load(f)
    with open('b_layer2.pik', 'rb') as f:  
        b2 = pickle.load(f)
    with open('b_layer3.pik', 'rb') as f:  
        b3 = pickle.load(f)
    with open('b_layer4.pik', 'rb') as f:  
        b4 = pickle.load(f)
    with open('b_layer5.pik', 'rb') as f:  
        b5 = pickle.load(f)
    with open('b_layer6.pik', 'rb') as f:  
        b6 = pickle.load(f)


    #Generator
    w_layer1 = tf.Variable(w1)
    w_layer2 = tf.Variable(w2)
    w_layer3 = tf.Variable(w3)
    w_layer4 = tf.Variable(w4)
    w_layer5 = tf.Variable(w5)
    w_layer6 = tf.Variable(w6)

    b_layer1 = tf.Variable(b1)
    b_layer2 = tf.Variable(b2)
    b_layer3 = tf.Variable(b3)
    b_layer4 = tf.Variable(b4)
    b_layer5 = tf.Variable(b5)
    b_layer6 = tf.Variable(b6)

    #Discriminator
    D_w_layer1 = tf.Variable(tf.random_normal([n_input, n_hidden_layer1]))
    D_w_layer2 = tf.Variable(tf.random_normal([n_hidden_layer1, n_hidden_layer2]))
    D_w_layer3 = tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer3]))
    D_w_layer4 = tf.Variable(tf.random_normal([n_hidden_layer3, 1]))

    D_b_layer1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer1, ]))
    D_b_layer2 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer2, ]))
    D_b_layer3 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer3, ]))
    D_b_layer4 = tf.Variable(tf.constant(0.1, shape=[1, ]))


    # with tf.Graph().as_default():

    train_GAN()

