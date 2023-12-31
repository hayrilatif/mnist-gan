from hyperparams import Params
from models import train_step,generate_sample,get_discriminator,get_generator
from dataset_creator import get_dataset

import tensorflow as tf


#Define tensorflow optimizers for weight updates.
optimizer_discriminator=tf.keras.optimizers.Adam(Params.discriminator_lr)
optimizer_generator=tf.keras.optimizers.Adam(Params.generator_lr)

#Get dataset.
train_data=get_dataset()

#Create generator and discriminator models.
generator,discriminator=get_generator(),get_discriminator()

#Define loss function.
loss_f=tf.keras.losses.Hinge()

disc_losses=[]
gen_losses=[]
for ep in range(Params.epochs):
    loss_d=[]
    loss_g=[]
    for x in train_data:
        loss=train_step(x,discriminator,generator,loss_f,optimizer_discriminator,optimizer_generator,Params.batchsize)
        loss_d.append(loss[0])
        loss_g.append(loss[1])
    
    disc_losses.append(sum(loss_d)/len(loss_d))
    gen_losses.append(sum(loss_g)/len(loss_g))
    
    print(f"Epoch: {ep+1}, DLoss: {disc_losses[-1]}, GLoss: {gen_losses[-1]}")