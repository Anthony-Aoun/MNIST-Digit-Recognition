from PIL import Image

target = 28

img = Image.open('../../input/test/three.png')
width = img.size[0]
height = img.size[1]

#vertical image
if width <= height:
	#resize
	wpercent = (target/float(width))
	hsize = int((float(height)*float(wpercent)))
	img = img.resize((target,hsize), Image.ANTIALIAS)
	#crop
	left, top, right, bottom = 0, ((hsize//2) - (target//2)), target, ((hsize//2) + (target//2))
	img = img.crop((left, top, right, bottom))
	#save
	img.save('result.png') 

#horizontal image
else:
	hpercent = (target/float(height))
	wsize = int((float(width)*float(hpercent)))
	img = img.resize((wsize,target), Image.ANTIALIAS)
	left, top, right, bottom = ((wsize//2) - (target//2)), 0, ((wsize//2) + (target//2)), target
	img = img.crop((left, top, right, bottom))
	img.save('result.png') 
