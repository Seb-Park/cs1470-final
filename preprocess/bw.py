import PIL.Image as Image
import sys
import os

INPUTS = 'photos'
SAVETO = 'photos_bw'
inputs = os.listdir(INPUTS)
try:
  inputs.remove('.DS_Store')
except: pass
PRINT_ON = int(len(inputs)*0.05) # print every 5%

for i in range(len(inputs)):
  inpt = inputs[i]
  fname = INPUTS+'/'+inpt
  image = Image.open(fname)
  out = image.convert('L') # ignore color data
  out.save(SAVETO+'/'+inpt)

  if (i+1) % PRINT_ON == 0 or i == 0 or i == len(inputs)-1:
    sys.stdout.write('\rConverting and saving images -- ')
    sys.stdout.write(str(int((i+1)/len(inputs)*100))+"%")
    sys.stdout.flush()
