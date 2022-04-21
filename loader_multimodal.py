import os
import random
import numpy as np
from skimage.measure import block_reduce
import nibabel as nib
from numpy.random import seed
import sys

#------------------------------ Setting seeds-------------------------------
random.seed(1234) 
seed(1234)
#---------------------------------------------------------------------------


class Data(object):
    
    def __init__(self, data_folder = "", scanner_names_to_load = None, 
                 downsample = False, normalize_volumes = False, downsample1 = 1, 
                 downsample2 = 1, Swap_axis = False, upsampling = False, mask_adr = ''):
        
        self.data_folder = data_folder
        self.scanner_names_to_load = scanner_names_to_load
        
        self.ge = None
        self.philips = None
        self.trio = None
        self.prisma = None
        self.mask = []

        self.refDict = {'ge': None, 'philips': None, 'trio': None, 'prisma': None}
        self.refAffineDict = {'ge': None, 'philips': None, 'trio': None, 'prisma': None}
        
        # I may want to delete these later. 
        self.downsampling = downsample
        self.norm_vols = normalize_volumes
        self.downsample1 = downsample1
        self.downsample2 = downsample2
        self.Swap_axis = Swap_axis
        self.upsampling = upsampling
        self.mask_adr = mask_adr
        self.mask_affine = np.array([])
        
        
    def swap_axis(self, data):
        
        for i, x in enumerate(data):
            data[i] = np.swapaxes(x, 0, 2)
            
            
    def downsamplImages(self, data, downsample1, downsample2):
        
        for i, x in enumerate(data):
            data[i] = block_reduce(x, block_size=(1, downsample1, downsample2), func=np.mean)
      
            
    def normalizeImages(self, data):
        
        for i, x in enumerate(data):
            data[i] = x / np.mean(x)
            
    
    
    def load_mask(self, mask_adr):
        
        if (mask_adr != ''):
            mask_img = nib.load(self.mask_adr)
            affine = mask_img.affine
            mask = mask_img.get_data()
        return mask, affine
    
    
    def check_loaded_data(self, no_images_of_scanners, All_file_names):
        
        '''
        This function
                      1. check "paired" data for each subject to have scans taken by all scanner. 
                      2. check "paired" data for each subject to have identical filenames across scanners. 
        '''
        
        # 1. check paired data for each subject to have scans taken by all scanner.
        if (len(set(no_images_of_scanners)) != 1):
            print("Error:The data is not paired. You might forgot image/images of one/some subjects.")
            print("Counts:")
            for ind in range(0, len(self.scanner_names_to_load)):
                print("Number of subjects for scanner ", self.scanner_names_to_load[ind], ": ", str(no_images_of_scanners[ind]))
            sys.exit()
            
        # 2. check paired data for each subject to have identical filenames across scanners.    
        No_of_subjects = no_images_of_scanners[0]
        No_file_names = len(set(All_file_names))
        if (No_file_names != No_of_subjects):
            print("Error: The data is not paired. There is mismatch of filenames across scanners. \n All images of the same should have identical file names across scanner! ")
            sys.exit()
            
            
    def load_scans_of_scanner(self, scanner):
        
        '''
        This function: 
                        1) Read the images.
                        2) Collect the affine of images.
        ''' 
        address_to_scanner = os.path.join(self.data_folder, scanner)
        file_names = [f for f in os.listdir(address_to_scanner) if ('nii' in f)]
        file_names.sort() # Sort the list of names
        images = [nib.load(os.path.join(address_to_scanner, f)) for f in file_names]
        affines = [image.affine for image in images]
        data = [img.get_data() for img in images]
        return data, affines, file_names


    def load(self):
        
        '''
        This function
                     1. Reads and collect scans of each scanner. 
                     2. Check the structure of the paired data. 
                     3. Read the brain mask
        '''
        
        no_images_of_scanners = []
        All_file_names = []
        
        # 1. Reads and collect scans of each scanner and collect them in All_data. 
        print("--------------------Loading Data---------------------")
        for scanner_name in self.scanner_names_to_load:
            print ('Start loading for scanner ' + scanner_name)
            scans, affines, file_names = self.load_scans_of_scanner(scanner_name)
            All_file_names = All_file_names + file_names
            self.refDict[scanner_name] = scans
            self.refAffineDict[scanner_name] = affines
            no_images_of_scanners.append(len(scans))
            print ('Finish loading for scanner ' + scanner_name)
            print("")


        # 2. Check whether we have complete paired data. 
        self.check_loaded_data(no_images_of_scanners, All_file_names)
        print("----------------------------------------------------")
        
        # 3. Read the brain mask
        if (self.mask_adr != ''):
            mask_data, mask_affine = self.load_mask(self.mask_adr)
            self.mask.append(mask_data)
            self.mask_affine = mask_affine
         
    
        
    def preprocessing(self):   
           
        # Swap axis
        if self.Swap_axis:
            for mod, data in self.refDict.items():
                print ('Swap axis in ' + mod)
                self.swap_axis(data)
              
            # swap axis for mask  
            if (self.mask_adr != ''):
                self.swap_axis(self.mask)

                
        # upsamling
        if self.upsampling:
            for mod, data in self.refDict.items():
                print ('Upsampling for debugging, not for real run ' + mod)
                self.upsample_image(data, 0, 20, 8)
                #self.upsample_image(data, 0, 4, 0)
                
            # upsampling mask
            if (self.mask_adr != ''):
                self.upsample_image(self.mask, 0, 20, 8)
                #self.upsample_image(self.mask, 0, 4, 0)
                print("")
            
        
        # Downsample:
        if self.downsampling:
            for mod, data in self.refDict.items():
                print ('Downsampling in ' + mod)
                self.downsamplImages(data, self.downsample1, self.downsample2)
                
            # Downsampling mask
            if (self.mask_adr != ''):
                self.downsamplImages(self.mask, self.downsample1, self.downsample2)
                
                  
        # Normalizing images:
        if self.norm_vols:
            for mod, data in self.refDict.items():
                print ('Normalizing images in ' + mod)
                self.normalizeImages(data)

        self.ge = self.refDict['ge']
        self.philips = self.refDict['philips']
        self.trio = self.refDict['trio']
        self.prisma = self.refDict['prisma']
        if (self.mask_adr != ''):
            self.mask = self.mask[0]
            
            
   
    def load_image(self, mask_adr):
        
        img = nib.load(mask_adr)
        data = img.get_data()
        return data

    
    def upsample_image(self, data, x_added, y_added, z_added):

        for i, img in enumerate(data):
            
            sahpe_x = img.shape[0]
            sahpe_y = img.shape[1]
            sahpe_z = img.shape[2]
        
            Additionalx1 = np.zeros((int(x_added / 2), sahpe_y, sahpe_z), dtype = 'float32')
            Additionalx2 = np.zeros((int(x_added / 2), sahpe_y, sahpe_z), dtype = 'float32')
            img = np.concatenate((Additionalx1, img), axis = 0)
            img = np.concatenate((img, Additionalx2), axis = 0)
            sahpe_x += x_added
            
            Additionaly1 = np.zeros((sahpe_x, int(y_added / 2), sahpe_z), dtype = 'float32')
            Additionaly2 = np.zeros((sahpe_x, int(y_added / 2), sahpe_z), dtype = 'float32')
            img = np.concatenate((Additionaly1, img), axis = 1)
            img = np.concatenate((img, Additionaly2), axis = 1)
            sahpe_y += y_added
            
            
            Additionalz1 = np.zeros((sahpe_x, sahpe_y, int(z_added / 2)), dtype = 'float32')
            Additionalz2 = np.zeros((sahpe_x, sahpe_y, int(z_added / 2)), dtype = 'float32')
            img = np.concatenate((Additionalz1, img), axis = 2)
            img = np.concatenate((img, Additionalz2), axis = 2)
            sahpe_z += z_added
            data[i] = img


    def select_for_ids(self, scanner, ids, as_array=True):
        
        # I have error in here when I change the data 
        data_ids = [self.refDict[scanner][i] for i in ids] 
        
        if as_array:
            data_ids_ar = np.concatenate(data_ids)
            if len(data_ids_ar.shape) < 4:
                data_ids_ar = np.expand_dims(data_ids_ar, axis=3)
                return data_ids_ar
        else:
            data_ids_ar = data_ids
            if len(data_ids_ar[0].shape) < 4:
                data_ids_ar = [np.expand_dims(d, axis=3) for d in data_ids]
                return data_ids_ar
            
