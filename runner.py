import os
import numpy as np
import json
from model import Multimodel
from mult_image_save_callback import ImageSaveCallback


class Experiment(object):
       
    def __init__(self, input_scanners, data, latent_dim = 6, spatial_transformer = False,
                 ind_outs = False, fuse_outs = False, loss_weights= None, batch_size = 4, step1_epoches = 100, step2_epoches = 100
                 ,W = 4, H = 4, mask = None, mask_affine = None, learning_rate = 0.01, output_adr = 'Output'):
    
        self.input_scanners = input_scanners
        self.output_scanners = input_scanners
        self.latent_dim = latent_dim
        self.data = data
        self.spatial_transformer = spatial_transformer
        self.ind_outs = ind_outs
        self.fuse_outs = fuse_outs #***** No
        assert ind_outs or fuse_outs
        self.mm = None
        self.loss_weight = {}
        self.original_loss_weight = loss_weights
        self.batch_size = batch_size
        self.epoches = step1_epoches
        self.step1_epoches = step1_epoches
        self.step2_epoches = step2_epoches
        self.W = W
        self.H = H
        self.mask = mask
        self.mask_affine = mask_affine
        self.learning_rate = learning_rate
        self.output_adr = output_adr
      

    def run(self):
 
        No_of_subjects = len(list(self.data.refDict.values())[0]) 
        Sbj_IDs = [sbj_id for sbj_id in range(0, No_of_subjects)] 
        self.Two_Steptep_Training(Sbj_IDs)
        self.save(self.output_adr)      
  
  
    def Two_Steptep_Training(self, Sbj_IDs):
        '''
        This function is section 3.2. Two-step Training for Harmonization in the paper. It has two steps:
            1. Step 1 (Section 3.2.1 Step 1: Embedding Learning in the paper)
            2. Step 2 (Section 3.2.2 Step 2: Harmonization in the paper)
        '''
         
        # 1. Step 1
        optim, out_dict, loss_weights = self.fitting_first_step(Sbj_IDs)
        
        # 2. Step 2
        # 2.1. Freeze model.
        self.freeze_model()
        
        # 2.2.Compile model.
        self.mm.model.compile(optimizer= optim, loss=out_dict, loss_weights=loss_weights)

        # 2.3. Fit model
        self.fitting_second_step(Sbj_IDs, self.mm.model)
        
  
    def fitting_first_step(self, Sbj_IDs, model=None):
        
        # Reset weights will be used for step-2
        self.loss_weight['C2_general_embedding'] = 0.0
        self.loss_weight['C3_embeddings_source_to_target'] = 0.0
        self.loss_weight['C4_synthesized_images'] = 0.0 # This is a dummy weight.
        
        # Set weights for Step1
        self.loss_weight['concat'] = self.original_loss_weight['lambda_1']
        self.loss_weight['C1_embeddings'] = self.original_loss_weight['lambda_2']
        

        # Learning Model
        if model is None:
            print('-----------------------------------------------------------------------')
            print('Creating model...')
            optim, out_dict, loss_weights = self.create_model()
        assert self.mm.model is not None

        # Add one channel to input data.
        train_in = [self.data.select_for_ids(scanner, Sbj_IDs) for scanner in self.input_scanners]
        
        # Add one channel to output data.
        train_out = [self.data.select_for_ids(scanner, Sbj_IDs) for scanner in self.output_scanners]
        
                
        # Add the target output for the similarity of the embeddings. 
        train_shape = (train_out[0].shape[0], train_out[0].shape[1], train_out[0].shape[2], train_out[0].shape[3])
        if len(self.input_scanners) > 1:
            train_out += [np.zeros(shape=train_shape) for i in range(2)]
            
        
        # Setting parameters for learning
        cb_train_in = [self.data.select_for_ids(scanner, Sbj_IDs, as_array=False) for scanner in self.input_scanners]
        cb_train_out = [self.data.select_for_ids(scanner, Sbj_IDs, as_array=False) for scanner in self.output_scanners]
                
        # Setting the outputs for saving.
        cb = ImageSaveCallback(cb_train_in, cb_train_out, self.output_adr,
                               self.output_scanners, self.input_scanners, 
                               self.mask_affine, self.step1_epoches, 'Model_Step1')
                             
        
        print('------------------------ Step1: Start fitting model------------------------')
        self.mm.model.fit(train_in, train_out, epochs = self.epoches, batch_size = self.batch_size,
                          callbacks=[cb])
        print('------------------------ Step1: Finished fitting model------------------------')
        
        return optim, out_dict, loss_weights
    
   
    def fitting_second_step(self, Sbj_IDs, model=None):

        # Reset weights were used in step-2
        self.loss_weight['C2_general_embedding'] = 0.0
        self.loss_weight['C3_embeddings_source_to_target'] = 0.0
        self.loss_weight['concat'] = 0.0
        
        # Set weights for Step 2
        self.loss_weight['C1_embeddings'] = self.original_loss_weight['lambda_3']
        self.loss_weight['C4_synthesized_images'] = self.original_loss_weight['lambda_4']

        # Learning Model
        if model is None:
            print('Creating model...')
            self.create_model()
        assert self.mm.model is not None

        
        # All Data, unit = slices.
        train_in = [self.data.select_for_ids(scanner, Sbj_IDs) for scanner in self.input_scanners]
        
        
        # Preparing target images for output: there's 1 output per embedding plus 1 output for the total variance embedding
        train_out = [self.data.select_for_ids(scanner, Sbj_IDs) for scanner in self.output_scanners]
        
        train_shape = (train_out[0].shape[0], train_out[0].shape[1], train_out[0].shape[2], train_out[0].shape[3])

        # Add the target output for the similarity of the embeddings. 
        if len(self.input_scanners) > 1:
            train_out += [np.zeros(shape=train_shape) for i in range(2)]
            
        # Setting parameters for learning
        cb_train_in = [self.data.select_for_ids(mod, Sbj_IDs, as_array=False) for mod in self.input_scanners]
        cb_train_out = [self.data.select_for_ids(mod, Sbj_IDs, as_array=False) for mod in self.output_scanners]
                
        # Setting the outputs for saving.
        cb = ImageSaveCallback(cb_train_in, cb_train_out, self.output_adr,
                               self.output_scanners, self.input_scanners,
                               self.mask_affine, self.step2_epoches, 'Model_Step2')
                           
        print('------------------------------ Retraining model ------------------------------')
        self.mm.model.fit(train_in, train_out, epochs = self.epoches, batch_size = self.batch_size,
                          callbacks=[cb])
        print('------------------------------Finish retraining model------------------------------')

    
    def freeze_model(self):
        '''
        This function freezes the decoders of the model. 
        '''
        for lay in self.mm.model.layers:
            if ('dec' not in lay.name):
                lay .trainable = False
                
              
    def get_similarity(self, cb, path_to_test, vol_num):
        
        results = cb.save_similarity_measures_for_data(cb.train_inputs, cb.train_outputs)
        cb.write_similarity_measures(path_to_test, os.path.join('end_sres_test_' + str(vol_num), 'sim_measure'), results)
        
        return results


    def create_model(self):

        mod = self.input_scanners[0]
        chn = self.data.select_for_ids(mod, [0]).shape[3] 

        self.mm = Multimodel(self.input_scanners, self.output_scanners, self.latent_dim, chn,
                             self.spatial_transformer, self.ind_outs, self.fuse_outs, self.loss_weight,
                             self.W, self.H)
        
        optim, out_dict, loss_weights = self.mm.build_linear_without_Embeded_Images_New_loss(self.learning_rate)
        return optim, out_dict, loss_weights

        
    def save(self, folder_name):
        
        print ('Saving experiment details')
        exp_json = {'input_scanners': self.input_scanners,
                    'latent_dim': self.latent_dim,
                    'model_layers': [l.name for l in self.mm.model.layers],
                    'encoder_params': [l.count_params() for l in self.mm.model.layers if
                                       'enc_' + self.input_scanners[0] in l.name]
                    }
        with open(folder_name + '/experiment_config.json', 'w') as f:
            json.dump(exp_json, f)

            

