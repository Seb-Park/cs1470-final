import PIL.Image as Image
import sys
import os

INPUTS = 'dataset'
SAVETO = 'inputs'
inputs = os.listdir(INPUTS)
try:
  inputs.remove('.DS_Store')
except: pass
PRINT_ON = int(len(inputs)*0.05) # print every 5%

for i in range(len(inputs)):
  inpt = inputs[i]
  fname = INPUTS+'/'+inpt
  image = Image.open(fname)
  width, height = image.size
  if width > height*2/3.0:
    # keep center (height*2/3) of the width
    out = image.crop((width/2.0 - height*1/3.0, 0, width/2.0 + height*1/3.0, height))
  elif width < height*2/3.0:
    # keep top (width*3/2) of the height
    out = image.crop((0, 0, width, width*3/2.0))
  out = out.resize((100, 150))
  out = out.convert('RGB') # ignore transparency data
  out.save(SAVETO+'/'+inpt)
  

  if i % PRINT_ON == 0:
    sys.stdout.write('\rResizing and saving images -- ')
    sys.stdout.write(str(int((i+1)/len(inputs)*100))+"%")
    sys.stdout.flush()