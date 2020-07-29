from utilities_function import utilities, step1_f, step2_f

'''
Implementation of the Zhang Suen Algorithm
'''

def zhang_suen(image):
    thinned_img = image.copy()
    step1 = step2 = 1
    
    length = thinned_img.shape[0]
    width = thinned_img.shape[1]
    
    white = 0
    black = 1
                    
    while step1 or step2:
      
        step1 = step2 = 0
         
        for i in range(1, length-1):
            for j in range(1, width-1):
                         
                px = thinned_img[i,j]
                
                A, B, n = utilities(i,j,thinned_img)
                
                if(px == black and step1_f(A,B,n)):
                    thinned_img[i,j] = white
                    step1 = 1
                    
                if(px == black and step2_f(A,B,n)):
                    thinned_img[i,j] = white
                    step2 = 1

    return thinned_img



    




