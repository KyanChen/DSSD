import tensorflow as tf

#  define common parameter
tf.app.flags.DEFINE_bool('isTraining','True','define if is train or test')
tf.app.flags.DEFINE_integer('num_classes','2','The number of classes')
tf.app.flags.DEFINE_integer('imgSize','512','image size')
tf.app.flags.DEFINE_string('dataSet',r'Tfrecords\Tail\positiveSamples20191118','')
tf.app.flags.DEFINE_string('logdir',r'log\tail','')
tf.app.flags.DEFINE_string('model_path','model\model_tail','')
tf.app.flags.DEFINE_string('reflectance_path',r'tfrecords','')
tf.app.flags.DEFINE_boolean('with_gan_argu', False,'whether or not with gan argument')
#  define training parameter
tf.app.flags.DEFINE_integer('iters','50000','')
tf.app.flags.DEFINE_float('learning_rate','0.0001','learning rate')
tf.app.flags.DEFINE_integer('batch_size','2','')
tf.app.flags.DEFINE_string('testDataSet',r'I:\Code\多源遥感大数据目标检测\Tfrecords\Tail\positiveSamples20191118','')
tf.app.flags.DEFINE_string('shelterDataSet',r'tfrecords','')

FLAGS=tf.app.flags.FLAGS

