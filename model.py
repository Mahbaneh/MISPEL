import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
sys.setrecursionlimit(10000)
from Linear_layer import Linear



class Multimodel(object):

    
    def __init__(self, input_modalities, output_modalities, latent_dim, channels, spatial_transformer,
                 ind_outs, fuse_outs, loss_weight, dim_H, dim_W):
        
        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.latent_dim = latent_dim
        self.channels = channels
        self.spatial_transformer = spatial_transformer
        self.ind_outs = ind_outs
        self.fuse_outs = fuse_outs
        self.num_emb = len(input_modalities) + 1
        self.loss_weight = loss_weight
        self.H, self.W = dim_H, dim_W  
        self.decoders = [] 


    def encoder_maker(self, modality):
        inp = keras.Input(shape = (self.H, self.W, self.channels), name='enc_' + modality + '_input')
        
        # Down1
        conv = layers.Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv1')(inp)
        act = layers.LeakyReLU()(conv)
        conv = layers.Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv2')(act)
        act1 = layers.LeakyReLU()(conv)        
        pool = layers.MaxPooling2D(pool_size=(2, 2))(act1) # downsample 1st level
        
        # Down2
        conv = layers.Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv3')(pool)
        act = layers.LeakyReLU()(conv)
        conv = layers.Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv4')(act)
        act2 = layers.LeakyReLU()(conv)        
        pool = layers.MaxPooling2D(pool_size=(2, 2))(act2) # downsample 2nd level
        
        # Down3 = Embedding
        conv = layers.Conv2D(128, 3, padding='same', name='enc_' + modality + '_conv5')(pool)
        act = layers.LeakyReLU()(conv)
        conv = layers.Conv2D(128, 3, padding='same', name='enc_' + modality + '_conv6')(act)
        act = layers.LeakyReLU()(conv)

        # Up1
        ups = layers.UpSampling2D(size=(2, 2))(act) # upsample 2nd level
        conv = layers.Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv7')(ups)
        skip = layers.concatenate([act2, conv], axis = 3, name='enc_' + modality + '_skip1')  
        conv = layers.Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv8')(skip)
        act = layers.LeakyReLU()(conv)
        conv = layers.Conv2D(64, 3, padding='same', name='enc_' + modality + '_conv9')(act)
        act = layers.LeakyReLU()(conv)

        # Up2
        ups = layers.UpSampling2D(size=(2, 2))(act)
        conv = layers.Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv10')(ups)
        skip = layers.concatenate([act1, conv], axis=3, name='enc_' + modality + '_skip2') 
        conv = layers.Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv11')(skip)
        act = layers.LeakyReLU()(conv)
        conv = layers.Conv2D(32, 3, padding='same', name='enc_' + modality + '_conv12')(act)
        act = layers.LeakyReLU()(conv)

        conv_ld = self.latent_dim
        conv = layers.Conv2D(conv_ld, 3, padding='same', name='enc_' + modality + '_conv13')(act)
        lr =layers.LeakyReLU()(conv)
        lr._name = name ='enc_' + modality + '_conv13'
        
        return inp, lr

    
    def Liner_decoder_maker(self, modality_in, modality_out):
        
        inp = keras.Input(shape=(self.H, self.W, self.latent_dim), name='dec_' + modality_in + '_' + modality_out + '_input')
        lin_layer = Linear(self.latent_dim, 'dec_' + modality_in + '_' + modality_out + '_conv5')
        conv = lin_layer(inp)
        model = keras.Model(inputs=inp, outputs=conv, name='decoder_' + modality_in + '_' + modality_out)
        return model


    def get_embedding_distance_outputs_for_linear(self, embeddings):
        
        if len(self.inputs) == 1:
            print( 'Skipping embedding distance outputs for unimodal model')
            return []
        
        outputs = list()

        ind_emb = embeddings[:]
        weighted_rep = embeddings[-1]

        all_emb_flattened = [new_flatten(emb) for emb in ind_emb]
        concat_emb = layers.concatenate(all_emb_flattened, axis=1, name='em_concat')
        concat_emb._name = 'em_concat'
        outputs.append(concat_emb)

        fused_emb = new_flatten(weighted_rep, name='em_fused')
        fused_emb._name = 'em_fused'
        outputs.append(fused_emb)
        
        return outputs
    
       
    def concatenate_synthesized_images(self, data):
        
        concatenated_data = tf.concat(data, 3)
        return concatenated_data
    

    
    def build_linear_without_Embeded_Images_New_loss(self, learning_rate):
        
        # 0. ignore general embedding
        self.num_emb = self.num_emb - 1
         
        print('Latent dimensions: ' + str(self.latent_dim))
        
        # 1. Build encoders.
        encoders = [self.encoder_maker(m) for m in self.input_modalities]
        
        # 2. Get the encoder's embedding. 
        ind_emb = [lr for (input, lr) in encoders]
        
        # 3. Save the encode's embeddings. 
        self.org_ind_emb = [lr for (input, lr) in encoders]
        
        # 4. Save the input of the encoders as the inputs of the final model to train. 
        self.inputs = [input for (input, lr) in encoders]

        # 5. deleted steps 5
        # Substitute step 6
        self.all_emb = ind_emb
        
        # 7. Build the decoder models and return them. 
        outputs = []

        for ind in range(0, len(self.output_modalities)):
            lin_decoders = []
            lin_mod = self.output_modalities[ind]
            lin_decoders = lin_decoders + [self.Liner_decoder_maker(lin_mod, lin_mod)]
            self.decoders = self.decoders + lin_decoders
            new_output = get_Linear_decoder_outputs_without_embeddedImages(lin_mod, lin_decoders, [self.all_emb[ind]], ind)
            outputs = outputs + new_output


        # 8.2. (For c2). This is another tensor for the similarity of the embeddings. 
        embedding_distance_temp = self.get_embedding_distance_outputs_for_linear(self.all_emb) 
        outputs += [embedding_distance_temp[0]]
        outputs += [self.concatenate_synthesized_images(outputs[0:len(self.output_modalities)])]
        
        # 9. Define functions for outputs (losses). 
        # 9.1. Define MAE for C1 and C3. 
        out_dict = {'em_%d_dec_%s' % (ind_dec, self.output_modalities[ind_dec]): mae for ind_dec in range(0, len(self.output_modalities))}
        
        # 9.2. Define Var function for C2.      
        if len(self.inputs) > 1:
                out_dict['em_concat'] = embedding_distance
                
        # 9.3. Add the general embedding to the list of outputs. 
        out_dict['tf_op_layer_concat'] = similarity_of_synthesized


        # 10 Generate loss weights.         
        # 10.1. Extracting weights from input.  
        get_indiv_weight = lambda loss_type: self.loss_weight[loss_type] if self.ind_outs else 0.0 #Function that get the 'input mod' and extract its output_weight from the code arguments.
        get_fused_weight = lambda loss_type: self.loss_weight[loss_type] if self.fuse_outs else 0.0 # Function that get the 'fuse mod' and extract its output_weight from the code arguments. 
        
        
        # 10.2. Generate the loss weights for  C1 and C3. 
        # 
        loss_weights = {}
        for ind in range(0, len(self.output_modalities)):
            loss_weights['em_%d_dec_%s' % (ind, self.output_modalities[ind])] = get_indiv_weight('C1_embeddings')

        # 10.3. Generate the loss weight for C2. 
        if len(self.inputs) > 1:
            loss_weights['em_concat'] = self.loss_weight['concat']
          
        # 10.4. Set the loss weight for the fusion which should be zero.   
        loss_weights['tf_op_layer_concat'] = self.loss_weight['C4_synthesized_images']

        # 11. Define the model: The inputs are the images and the outputs are the 
        self.model = keras.Model(inputs=self.inputs, outputs=outputs)
        
        # 12. Compile the network: the loss are the loss functions and loss_weights are the weights for losses. 
        optim = keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer= optim, loss=out_dict, loss_weights=loss_weights)

        print("Model complied!")
        return optim, out_dict, loss_weights
    

def get_Linear_decoder_outputs_without_embeddedImages(output_modalities, decoders, embeddings, input_index):
    
    outputs = list()
    for ind in range(0, len(embeddings)):
        out_em = decoders[ind](embeddings[ind])
        name = 'em_' + str(input_index) + '_dec_' + output_modalities
        l = layers.Lambda(lambda x: x + 0, name=name)(out_em)
        l._name = name
        outputs.append(l)

    return outputs


def embedding_distance(y_true, y_pred):
    return K.var(y_pred, axis=1)


def new_flatten(emb, name=''): 
    l = layers.Lambda(lambda x: K.batch_flatten(x))(emb) # Flatten data for each batch
    l = layers.Lambda(lambda x: K.expand_dims(x, axis=1), name=name)(l) # Addd dimaention at axix 1.
    return l


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def similarity_of_synthesized(y_true, y_pred):
    
    no_of_images = y_pred.shape[-1]
    sum = 0
    
    for indi in range(0, no_of_images):
        for indj in range(indi + 1, no_of_images):
            sum += mae(y_pred[:,:,:,indi], y_pred[:,:,:,indj])
    avg =  (sum * 1.0)/ no_of_images  
    return avg

