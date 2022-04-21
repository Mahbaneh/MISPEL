import os
from tensorflow.keras.callbacks import Callback
import numpy as np
import nibabel as nib


class ImageSaveCallback(Callback):

    def __init__(self, train_inputs, train_outputs, folder_name, output_scanners,
                  input_scanners, mask_affine, epoches, model_name = ''):
        super(ImageSaveCallback, self).__init__()

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.folder_name = folder_name
        self.output_scanners = output_scanners
        self.input_scanners = input_scanners
        self.epoches = epoches
        self.model_name = model_name
        self.affine = mask_affine

     
    def SaveAllSynthesizedImages_without_embeddedImages(self, filename):
        # 1. Reshape all data.
        train_sbj_no = len(self.train_inputs[0])
        modality_no = len(self.train_inputs)
        All_data = []
        
        for indi in range(0, train_sbj_no):
            sbj_data = []
            for indj in range(0, modality_no):
                sbj_data = sbj_data + [self.train_inputs[indj][indi]]
            
            All_data.append(sbj_data)
        
        # 2. Make the Nifti folder. 
        if "Nifti" in os.listdir(filename):
            adr = os.path.join(filename, 'Nifti')
        else:
            adr = os.path.join(filename, 'Nifti')
            os.makedirs(adr)
        
        # making image names 
        image_nmaes = []
        for out_mod in self.output_scanners:
            image_nmaes.append(out_mod + '_' + out_mod)
          
            
        # 3. Predict output 
        for sbj_ind in range(0, len(All_data)):
            model_output_for_Subject_no = self.model.predict(All_data[sbj_ind])
            
            image_adr = os.path.join(adr, 'image_' + str(sbj_ind))
                
            if image_adr in os.listdir(adr):
                pass
            else:
                os.makedirs(image_adr)

            # swap the image again to match with affine. 
            synthesized_images = [np.swapaxes(syn_image, 2, 0) for syn_image in model_output_for_Subject_no]
            
            for ind in range(0, len(self.output_scanners)):                
                # write image
                image_adr = os.path.join(adr, 'image_' + str(sbj_ind) ,'syn_image_' + image_nmaes[ind] + '.nii.gz') 
                
                # write the image with hard_coded affine.
                Synthesized_image = nib.Nifti1Image(synthesized_images[ind], self.affine)
                nib.save(Synthesized_image, image_adr)
     
                

    def on_epoch_end(self, epoch, logs = None):

        # save the model. 
        if logs is None:
            logs = {}
        
        model_dir_address = os.path.join(self.folder_name , self.model_name + '_%d' % epoch)
        if (self.epoches == epoch + 1):
            # save the model
            self.model.save_weights(model_dir_address + '/')
            
            # save all synthesized Images. 
            self.SaveAllSynthesizedImages_without_embeddedImages(model_dir_address)

                
