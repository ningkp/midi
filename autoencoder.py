import tensorflow as tf
import midi
import numpy as np
import os
 

def app_run(_epochs,_batch_size,_lr):

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
	 
	 
	def to_sample(probs):
		return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
	 
	def gibbs_sample(k):
		def body(count, k, xk):
			hk = to_sample(tf.sigmoid(tf.matmul(xk, W) + bh))
			xk = to_sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
			return count+1, k, xk
	 
		count = tf.constant(0)
		def condition(count,  k, xk):
			print("jinlai")
			return count < k
		[_, _, x_sample] = tf.while_loop(condition, body, [count, tf.constant(k), X])
	 
		x_sample = tf.stop_gradient(x_sample) 
		return x_sample
	 
	def neural_network():
		global W
		W  = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01))
		global bh
		bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32))
		global bv
		bv = tf.Variable(tf.zeros([1, n_input],  tf.float32))
	 
		x_sample = gibbs_sample(1)
		print("x_sample:" + str(x_sample.shape))
		h = to_sample(tf.sigmoid(tf.matmul(X, W) + bh))
		print("h:" + str(h.shape))
		h_sample = to_sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))
	 	print("h_sample:" + str(h_sample.shape))

		learning_rate = tf.constant(0.005, tf.float32)
		size_bt = tf.cast(tf.shape(X)[0], tf.float32)
		W_adder  = tf.multiply(learning_rate/size_bt, tf.subtract(tf.matmul(tf.transpose(X), h), tf.matmul(tf.transpose(x_sample), h_sample)))
		bv_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(X, x_sample), 0, True))
		bh_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
		update = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]
		print(np.array(update).shape)
		return update
	 
	def train_neural_network():
		update = neural_network()
		creat_music = []
		flag = False

		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())
	 
			saver = tf.train.Saver(tf.all_variables())
	 
			epochs = 256
			batch_size = 64
			for epoch in range(epochs):
				print(epoch)
				for song in songs:
					song = np.array(song)
					# print(song.shape)
					song = song[:int(np.floor(song.shape[0]/n_timesteps) * n_timesteps)]
					song = np.reshape(song, [song.shape[0]//n_timesteps, song.shape[1] * n_timesteps])
					# print(song.shape)
					# return
					for i in range(0, len(song), batch_size): 
						train_x = song[i:i+batch_size]
						if flag == False:
							for k in range(0,len(train_x)):
								creat_music.append(train_x[k])
								print("coming!!!")
							flag = True
							print(creat_music)
							print(np.array(creat_music).shape)
						sess.run(update, feed_dict={X: train_x})

				if epoch == epochs - 1:
					saver.save(sess, 'midi.module')
	 		
	 		print(np.array(creat_music).shape)
			sample = gibbs_sample(1).eval(session=sess, feed_dict={X: creat_music})
			S = np.reshape(sample[0,:], (n_timesteps, 2 * note_range))
			noteStateMatrixToMidi(S, "auto_gen_music")
			print('creat auto_gen_music.mid  file')

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
				# print(song.shape)	  #(2,19968)
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
		noteStateMatrixToMidi(S, "auto_gen_music")
		print('creat auto_gen_music.mid  file')
		return



	note_range = upper_bound - lower_bound
	n_timesteps = 128
	n_input = 2 * note_range * n_timesteps
	epochs = _epochs
	batch_size = _batch_size
	lr = _lr

	n_hidden_layer1 = 128
	n_hidden_layer2 = 64
	n_hidden_layer3 = 32
	n_hidden_layer4 = 64
	n_hidden_layer5 = 128
	n_output = n_input


	X = tf.placeholder(tf.float32, [None, n_input])
	Y = tf.placeholder(tf.float32, [None, n_output])


	w_layer1 = tf.Variable(tf.random_normal([n_input, n_hidden_layer1]))
	w_layer2 = tf.Variable(tf.random_normal([n_hidden_layer1, n_hidden_layer2]))
	w_layer3 = tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer3]))
	w_layer4 = tf.Variable(tf.random_normal([n_hidden_layer3, n_hidden_layer4]))
	w_layer5 = tf.Variable(tf.random_normal([n_hidden_layer4, n_hidden_layer5]))
	w_layer6 = tf.Variable(tf.random_normal([n_hidden_layer5, n_output]))

	b_layer1 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer1, ]))
	b_layer2 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer2, ]))
	b_layer3 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer3, ]))
	b_layer4 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer4, ]))
	b_layer5 = tf.Variable(tf.constant(0.1, shape=[n_hidden_layer5, ]))
	b_layer6 = tf.Variable(tf.constant(0.1, shape=[n_output, ]))




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
	    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

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