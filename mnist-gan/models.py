import tensorflow as tf
import numpy as np

#Define a convolutional block that contains dropout and activation within it for ease of use.
class Conv2DBlock(tf.keras.layers.Layer):
    def __init__(self,filter_count,filter_size,activation,dropout_rate):
        super(Conv2DBlock,self).__init__()
        self.filter_count=filter_count
        self.filter_size=filter_size
        self.activation=activation
        self.dropout_rate=dropout_rate
        
        self.conv=tf.keras.layers.Conv2D(self.filter_count,self.filter_size,padding="same")
        self.bnorm=tf.keras.layers.BatchNormalization()
        self.act=tf.keras.layers.Activation(self.activation)
        self.dropo=tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self,x):
        conv=self.conv(x)
        act=self.act(conv)
        dropo=self.dropo(act)
        return dropo
    
#Define a transpose-convolutional block that contains dropout and activation within it for ease of use. This will used in up-sampling bundles.
class TransposeConv2DBlock(tf.keras.layers.Layer):
    def __init__(self,filter_count,filter_size,activation,dropout_rate,strides):
        super(TransposeConv2DBlock,self).__init__()
        self.filter_count=filter_count
        self.filter_size=filter_size
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.strides=strides
        
        self.conv=tf.keras.layers.Conv2DTranspose(self.filter_count,self.filter_size,strides=self.strides,padding="same")
        self.bnorm=tf.keras.layers.BatchNormalization()
        self.act=tf.keras.layers.Activation(self.activation)
        self.dropo=tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self,x):
        conv=self.conv(x)
        act=self.act(conv)
        dropo=self.dropo(act)
        return dropo

#This defines the downsample part of the discriminator network.
def downsample_bundle(filter_count,filter_size,activation,dropout_rate):
    def func(x):
        cb1=Conv2DBlock(filter_count,filter_size,activation,dropout_rate)(x)
        cb2=Conv2DBlock(filter_count,filter_size,activation,dropout_rate)(cb1)
        pool1=tf.keras.layers.MaxPooling2D()(cb2)
        return pool1
    return func

#This defines the upsample part of the generator network.
def upsample_bundle(filter_count,filter_size,activation,dropout_rate,strides):
    def func(x):
        cb1=TransposeConv2DBlock(filter_count,filter_size,activation,dropout_rate,1)(x)
        cb2=TransposeConv2DBlock(filter_count,filter_size,activation,dropout_rate,strides)(cb1)
        return cb2
    return func

#Build discriminator model.
def get_discriminator():
    inp=tf.keras.layers.Input((28,28,1))
    rs=tf.keras.layers.Resizing(32,32)(inp)
    
    db1=downsample_bundle(8,(3,3),"leaky_relu",0.)(rs)
    db2=downsample_bundle(16,(3,3),"leaky_relu",0.)(db1)
    db3=downsample_bundle(32,(3,3),"leaky_relu",0.)(db2)
    db4=downsample_bundle(64,(3,3),"leaky_relu",0.)(db3)
    gp=tf.keras.layers.GlobalMaxPooling2D()(db4)
    flat=tf.keras.layers.Dense(1)(gp)
    
    return tf.keras.models.Model(inp,flat)

#Build generator model.
def get_generator():
    inp=tf.keras.layers.Input((100))
    
    nd=tf.keras.layers.Dense(8*8*128)(inp)
    reshaped_tensor=tf.keras.layers.Reshape((8,8,128))(nd)
    
    ub1=upsample_bundle(32,(5,5),"leaky_relu",0.,2)(reshaped_tensor)
    ub2=upsample_bundle(32,(5,5),"leaky_relu",0.,2)(ub1)
    ub3=upsample_bundle(32,(5,5),"leaky_relu",0.,2)(ub2)
    
    last=tf.keras.layers.Conv2D(1,(1,1),padding="same",activation="sigmoid")(ub2)
    
    rs=tf.keras.layers.Resizing(28,28)(last)
    
    return tf.keras.models.Model(inp,rs)

#A method for generating data from normal distribution using generator model.
def generate_sample(generator):
    return generator.predict(np.random.normal(size=(100))[np.newaxis,...])[0]

#A method for adversarial training step.
@tf.function()
def train_step(x,discriminator,generator,loss_function,disc_opt,gen_opt,batchsize):
    with tf.GradientTape(persistent=True) as tape:
        #Discriminator update phase.
        dp_noise=tf.random.normal((batchsize,100))
        dp_fake_samples=generator(dp_noise)
        
        discrimination_real=discriminator(x)
        discrimination_fake=discriminator(dp_fake_samples)

        discriminator_loss_real_contribution=loss_function(tf.ones_like(discrimination_real),discrimination_real)
        discriminator_loss_fake_contribution=loss_function(tf.zeros_like(discrimination_fake),discrimination_fake)

        total_discriminator_loss=discriminator_loss_real_contribution+discriminator_loss_fake_contribution
        
        #Generator update phase.
        gp_noise=tf.random.normal((batchsize,100))
        gp_fake_samples=generator(gp_noise)
        
        gp_discrimination_fake=discriminator(gp_fake_samples)
        
        generator_loss=loss_function(tf.ones_like(gp_discrimination_fake),gp_discrimination_fake)
        
        
    discriminator_gradient=tape.gradient(total_discriminator_loss,discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(discriminator_gradient,discriminator.trainable_variables))
    
    generator_gradient=tape.gradient(generator_loss,generator.trainable_variables)
    gen_opt.apply_gradients(zip(generator_gradient,generator.trainable_variables))
    
    return tf.reduce_mean(total_discriminator_loss),tf.reduce_mean(generator_loss)