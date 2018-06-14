import tensorflow as tf
import numpy as np
import math

class Seq2Seq:
    PAD = 1
    EOS = 0
    logs_path = "./logs/"

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.input_size = 10
        self.encoder_cell = tf.contrib.rnn.LSTMCell(10)
        self.decoder_cell = tf.contrib.rnn.LSTMCell(20)
        self.decoder_hidden_units = self.decoder_cell.output_size

        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))
            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD
            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1
            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets
            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

        with tf.variable_scope("Embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.input_size],
                initializer=initializer,
                dtype=tf.float32)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_inputs)
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix,
                                                                        self.decoder_train_inputs)

        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
            )
            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, tf.contrib.rnn.LSTMStateTuple):
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

            (attention_keys,
             attention_values,
             attention_score_fn,
             attention_construct_fn) = tf.contrib.seq2seq.prepare_attention(
                attention_states=attention_states,
                attention_option="bahdanau",
                num_units=self.decoder_hidden_units,
            )

            decoder_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
                encoder_state=self.encoder_state,
                attention_keys=attention_keys,
                attention_values=attention_values,
                attention_score_fn=attention_score_fn,
                attention_construct_fn=attention_construct_fn,
                name='attention_decoder'
            )

            decoder_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self.encoder_state,
                attention_keys=attention_keys,
                attention_values=attention_values,
                attention_score_fn=attention_score_fn,
                attention_construct_fn=attention_construct_fn,
                embeddings=self.embedding_matrix,
                start_of_sequence_id=self.EOS,
                end_of_sequence_id=self.EOS,
                maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                num_decoder_symbols=self.vocab_size,
            )

        (self.decoder_outputs_train,
         self.decoder_state_train,
         self.decoder_context_state_train) = (
            tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_train,
                inputs=self.decoder_train_inputs_embedded,
                sequence_length=self.decoder_train_length,
                time_major=True,
                scope=scope,
            )
        )

        self.decoder_logits_train = output_fn(self.decoder_outputs_train)
        self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

        scope.reuse_variables()

        (self.decoder_logits_inference,
         self.decoder_state_inference,
         self.decoder_context_state_inference) = (
            tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_inference,
                time_major=True,
                scope=scope,
            )
        )
        self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1,
                                                      name='decoder_prediction_inference')

        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets,
                                                     weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        tf.summary.scalar("loss", self.loss)
        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

    def batch(self, inputs, max_sequence_length=None):
        """
        Args:
            inputs:
                list of sentences (integer lists)
            max_sequence_length:
                integer specifying how large should `max_time` dimension be.
                If None, maximum sequence length would be used

        Outputs:
            inputs_time_major:
                input sentences transformed into time-major matrix
                (shape [max_time, batch_size]) padded with 0s
            sequence_lengths:
                batch-sized list of integers specifying amount of active
                time steps in each input sequence
        """

        sequence_lengths = [len(seq) for seq in inputs]
        batch_size = len(inputs)

        if max_sequence_length is None:
            max_sequence_length = max(sequence_lengths)

        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element

        # [batch_size, max_time] -> [max_time, batch_size]
        inputs_time_major = inputs_batch_major.swapaxes(0, 1)

        return inputs_time_major, sequence_lengths

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = self.batch(input_seq)
        targets_, targets_length_ = self.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def make_inference_inputs(self, input_seq):
        inputs_, inputs_length_ = self.batch(input_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }

    def train(self, session, batches, batches_in_epoch=10, training_epochs=10, verbose=True, decoder=lambda x: x):
        loss_track = []
        batch_num = 0
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
        try:
            for epoch in range(training_epochs):
                for batch in batches:
                    train_seq_X, train_seq_Y = batch
                    fd = self.make_train_inputs(train_seq_X, train_seq_Y)
                    _, l, summary = session.run([self.train_op, self.loss, self.merged_summary_op], fd)
                    loss_track.append(l)
                    if verbose:
                        if batch_num == 0 or batch_num % batches_in_epoch == 0:
                            print('minibatch loss:\t{}'.format(session.run(self.loss, fd)))
                            for i, (e_in, dt_pred) in enumerate(zip(
                                    fd[self.encoder_inputs].T,
                                    session.run(self.decoder_prediction_train, fd).T
                            )):
                                print('sample\t{}:'.format(i + 1))
                                print('enc input\t{}'.format(decoder(e_in)))
                                print('dec train predicted\t{}'.format(decoder(dt_pred)))
                                if i >= 2:
                                    break
                            print()
                    batch_num += 1
                    summary_writer.add_summary(summary, epoch)
                saver.save(session, 'ckpt/cornell-model')
        except KeyboardInterrupt:
            print('training interrupted')


