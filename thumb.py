from PIL import Image
import glob, os

size = 43, 43

for infile in glob.glob("*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(file + "3.jpg", "JPEG")



for infile in glob.glob("*.jpg"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im = im.crop((0,0,32,32))
  
    im.save(file + "2.jpg", "JPEG")