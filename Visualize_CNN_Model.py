from keras.models import Model,load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array

def Visualize_CNN_Model():
    # -------------------------------------------------------------------------
    #                        Visualize CNN Model 
    # -------------------------------------------------------------------------
    # load model
    model = load_model('medical_diagnosis_cnn_model.h5')
    
    # load the image    
    img = load_img("IM-0009-0001.jpeg", target_size=(224, 224))              
    
    # convert to array
    img = img_to_array(img)

    # reshape and scale image
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')    
    img = img / 255             

    # create a new CNN activation_model
    layer_outputs = [layer.output for layer in model.layers[:10]]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # pass image through CNN activation_model 
    activations = activation_model.predict(img)

    # Vislalize intermediate layre    
    Display_Activation_Layers(activations, 4, 2, 3)
    
    Display_Activation_Layers(activations, 5, 2, 7)
 
    Display_Activation_Layers(activations, 7, 2, 9)
           
    
    
    
def Display_Activation_Layers(activations, col_size, row_size, act_index): 
    
    activation = activations[act_index]
    
    activation_index=0
    
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    
    for row in range(0,row_size):
    
        for col in range(0,col_size):
        
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            
            ax[row][col].axis('off')
            
            activation_index += 1             
 
# main entry            
Visualize_CNN_Model()

