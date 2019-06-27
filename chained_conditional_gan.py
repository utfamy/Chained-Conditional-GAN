import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 옵션 설정
######
total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28 * 28
n_noise = 128
n_class = 10

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

# MNIST 학습 시 사용되는 noise
Z_y = tf.placeholder(tf.float32, [None, n_noise])
# 2-hot encoded G2를 학습 시 사용되는 noise
Z_a = tf.placeholder(tf.float32, [None, n_noise])
# 2-hot encoded conditional vector
A = tf.placeholder(tf.float32, [None, n_class])


#2-hot encoded conditional vector 생성 함수
def get_a(size):
    return np.tile(np.array([[1,1,0,0,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0,0], [1,0,0,1,0,0,0,0,0,0], [1,0,0,0,1,0,0,0,0,0], [1,0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,1,0,0,0], [1,0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,1,0], [1,0,0,0,0,0,0,0,0,1], 
                             [0,1,1,0,0,0,0,0,0,0], [0,1,0,1,0,0,0,0,0,0], [0,1,0,0,1,0,0,0,0,0], [0,1,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,0,1,0,0], [0,1,0,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,0,0,1], 
                             [0,0,1,1,0,0,0,0,0,0], [0,0,1,0,1,0,0,0,0,0], [0,0,1,0,0,1,0,0,0,0], [0,0,1,0,0,0,1,0,0,0], [0,0,1,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0,0,1],
                             [0,0,0,1,1,0,0,0,0,0], [0,0,0,1,0,1,0,0,0,0], [0,0,0,1,0,0,1,0,0,0], [0,0,0,1,0,0,0,1,0,0], [0,0,0,1,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,0,1],
                             [0,0,0,0,1,1,0,0,0,0], [0,0,0,0,1,0,1,0,0,0], [0,0,0,0,1,0,0,1,0,0], [0,0,0,0,1,0,0,0,1,0], [0,0,0,0,1,0,0,0,0,1],
                             [0,0,0,0,0,1,1,0,0,0], [0,0,0,0,0,1,0,1,0,0], [0,0,0,0,0,1,0,0,1,0], [0,0,0,0,0,1,0,0,0,1],
                             [0,0,0,0,0,0,1,1,0,0], [0,0,0,0,0,0,1,0,1,0], [0,0,0,0,0,0,1,0,0,1],
                             [0,0,0,0,0,0,0,1,1,0], [0,0,0,0,0,0,0,1,0,1],
                             [0,0,0,0,0,0,0,0,1,1]]), (int(size/10),1))

def generator1(noise, labels, reuse=None):
    with tf.variable_scope('generator1') as scope:
        if reuse:
            scope.reuse_variables()
      
        # noise 값에 labels 정보를 추가합니다.
        inputs = tf.concat([noise, labels], 1)

        # TensorFlow 에서 제공하는 유틸리티 함수를 이용해 신경망을 매우 간단하게 구성할 수 있습니다.
        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input,
                                 activation=tf.nn.sigmoid)

    return output

def generator2(noise, labels, reuse=None):
    with tf.variable_scope('generator2') as scope:
        if reuse:
            scope.reuse_variables()
        
        # noise 값에 labels 정보를 추가합니다.
        inputs = tf.concat([noise, labels], 1)

        # TensorFlow 에서 제공하는 유틸리티 함수를 이용해 신경망을 매우 간단하게 구성할 수 있습니다.
        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input,
                                 activation=tf.nn.sigmoid)

    return output


def discriminator1(inputs, reuse=None):
    with tf.variable_scope('discriminator1') as scope:
        # 노이즈에서 생성한 이미지와 실제 이미지를 판별하는 모델의 변수를 동일하게 하기 위해,
        # 이전에 사용되었던 변수를 재사용하도록 합니다.
        if reuse:
            scope.reuse_variables()

        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        #output은 input의 10가지 클래스(0~9)를 구분하도록 1*10 벡터로 지정
        output = tf.layers.dense(hidden, 10,
                                 activation=None)

    return output

def discriminator2(inputs, reuse=None):
    with tf.variable_scope('discriminator2') as scope:
        # 노이즈에서 생성한 이미지와 실제 이미지를 판별하는 모델의 변수를 동일하게 하기 위해,
        # 이전에 사용되었던 변수를 재사용하도록 합니다.
        if reuse:
            scope.reuse_variables()

        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        #output은 input의 10가지 클래스(0~9)를 구분하도록 1*10 벡터로 지정
        output = tf.layers.dense(hidden, 10,
                                 activation=None)

    return output
  

def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])

# 생성 모델과 판별 모델에 Y 즉, labels 정보를 추가하여
# labels 정보에 해당하는 이미지를 생성할 수 있도록 유도합니다.


#phase 1 - G1이 MNIST data를 학습하도록 훈련
#D1은 G1과 MNIST data를 구분하고 동시에 MNIST data의 클래스를 판별
G1_1h = generator1(Z_y, Y)

D1_mn = discriminator1(X)
D1_g1 = discriminator1(G1_1h, True)

#phase 2 - G2가 2-hot encoded G1을 학습하도록 훈련 
#D2는 2-hot encoded G1과 2-hot encoded G2를 구분하고 2-hot encoded G1의 클래스를 판별(2-hot)
G1_2h = generator1(Z_a, A, True)
G2_2h = generator2(Z_a, A)

D2_g1 = discriminator2(G1_2h)
D2_g2 = discriminator2(G2_2h, True)

#phase 3 - 1-hot encoded G2의 성능 확인
G2_1h = generator2(Z_y, Y, True)
D1_g2 = discriminator1(G2_1h, True)

# 손실함수는 다음을 참고하여 GAN 논문에 나온 방식과는 약간 다르게 작성하였습니다.
# http://bamos.github.io/2016/08/09/deep-completion/
# 진짜 이미지를 판별하는 D_real 값은 1에 가깝도록,
# 가짜 이미지를 판별하는 D_gene 값은 0에 가깝도록 하는 손실 함수입니다.


#phase 1

loss_D1_mn = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D1_mn, labels=Y))
loss_D1_g1 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D1_g1, labels=tf.zeros_like(D1_g1)))

# MNIST의 클래스 구분 시 발생한 loss와 G1을 가짜로 판별할 때 생긴 loss를 더한 뒤 이 값을 최소화 하도록 최적화합니다.
loss_D1 = loss_D1_mn + loss_D1_g1


# G1을 MNIST에 가깝게 만들도록 학습시키기 위해, D1_g1 을 최대한 1에 가깝도록 만드는 손실함수입니다.
loss_G1 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D1_g1, labels=Y))

#phase 2

loss_D2_g1 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D2_g1, labels=A))
loss_D2_g2 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D2_g2, labels=tf.zeros_like(D2_g2)))

# 2-hot encoded G1의 클래스 구분 시 발생한 loss와 G2를 G1으로 판별할 때 생긴 loss를 더한 뒤 이 값을 최소화 하도록 최적화합니다.
loss_D2 = loss_D2_g1 + loss_D2_g2 

# 2-hot encoded G2를 2-hot encoded G1에 가깝게 만들도록 학습시키기 위해, D2_g2를 최대한 conditional vector에 가깝도록 만드는 손실함수입니다.
loss_G2 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D2_g2, labels=A))

#phase 3
# 1-hot encoded G2를 D1에서 판별했을 때 발생하는 loss를 측정합니다.
loss_D1_phase3 = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D1_g2, labels=Y))



# discriminator 와 generator scope 에서 사용된 변수들을 가져오기
vars_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='discriminator1')
vars_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='discriminator2')
vars_G1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='generator1')
vars_G2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='generator2')

# 모델 학습
train_D1 = tf.train.AdamOptimizer().minimize(loss_D1,
                                            var_list=vars_D1)
train_D2 = tf.train.AdamOptimizer().minimize(loss_D2,
                                            var_list=vars_D2)
train_G1 = tf.train.AdamOptimizer().minimize(loss_G1,
                                            var_list=vars_G1)
train_G2 = tf.train.AdamOptimizer().minimize(loss_G2,
                                            var_list=vars_G2)

# 변수 저장
saver = tf.train.Saver()


# 신경망 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())



#  변수 불러오기
ckpt = tf.train.get_checkpoint_state('./save/')
if tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
  saver.restore(sess, ckpt.model_checkpoint_path)
  print("variable is restored")

print(sess.run(vars_G1))



total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D1, loss_val_G1, loss_val_D2, loss_val_G2, loss_val_D1_phase3 = 0, 0, 0, 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise_y = get_noise(batch_size, n_noise)
        noise_a = get_noise(int(45*batch_size/10), n_noise)
        a = get_a(batch_size)

        _, loss_val_D1 = sess.run([train_D1, loss_D1],
                                 feed_dict={X: batch_xs, Y: batch_ys, Z_y: noise_y})
        _, loss_val_D2 = sess.run([train_D2, loss_D2],
                                  feed_dict={X: batch_xs, Z_a: noise_a, A: a})
        _, loss_val_G1 = sess.run([train_G1, loss_G1],
                                 feed_dict={Y: batch_ys, Z_y: noise_y})
        _, loss_val_G2 = sess.run([train_G2, loss_G2],
                                 feed_dict={Z_a: noise_a, A: a})
        loss_val_D1_phase3 = sess.run([loss_D1_phase3],
                                      feed_dict={Y: batch_ys, Z_y: noise_y})

    print('Epoch:', '%04d' % epoch,
          'D1 loss: {:.4}'.format(loss_val_D1),
          'D2 loss: {:.4}'.format(loss_val_D2),
          'G1 loss: {:.4}'.format(loss_val_G1),
          'G2 loss: {:.4}'.format(loss_val_G2),
          'Final loss: {}'.format(loss_val_D1_phase3)
         )

    #########
    # 학습이 되어가는 모습을 보기 위해 주기적으로 레이블에 따른 이미지를 생성하여 저장
    ######
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise_y = get_noise(sample_size, n_noise)
        noise_a = get_noise(int(45*sample_size/10), n_noise)
        a = get_a(sample_size)
        samples1_1h, result1_g1, samples1_2h, result2_g1, samples2_2h, result2_g2, samples2_1h, result1_g2 = sess.run([G1_1h, D1_g1, G1_2h, D2_g1, G2_2h, D2_g2, G2_1h, D1_g2],
                           feed_dict={Y: np.array([[1,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,1]]),
                                      Z_y: noise_y, Z_a: noise_a, A: a})
        print('D1_g1: {}'.format(result1_g1))
        print('D2_g1: {}'.format(result2_g1))
        print('D2_g2: {}'.format(result2_g2))
        print('D1_g2: {}'.format(result1_g2))
        
        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples1_1h[i], (28, 28)))

        plt.savefig('samples1_1h/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        
        
        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples1_2h[i], (28, 28)))

        plt.savefig('samples1_2h/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        saver.save(sess, './save/save.ckpt', global_step = epoch)
        
        
        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples2_2h[i], (28, 28)))

        plt.savefig('samples2_2h/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        
        
        
        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples2_1h[i], (28, 28)))

        plt.savefig('samples2_1h/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        
        
print('최적화 완료!')