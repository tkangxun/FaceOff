from adversarial_face_recognition import *
import os
import torch

## Standard normalization for ImageNet images found here:
## https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
apply = Applier()
## Transformations to be used later
tensorize = transforms.ToTensor()
imagize = transforms.ToPILImage()
## FaceNet PyTorch model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

## Image preprocessing
input_image_location =  './results/algo_example/me3.jpg'
target_image_location = './faces/target/chris.jpg'
input_test_location =   './results/algo_example/me2.jpg'
target_test_location =  './faces/target_tests/chris2.jpg'
physical_image_location ='./results/algo_example/me3_phone.jpg'

input_image = detect_face(input_image_location)[0]
print("\nInput detected and aligned")
target_image = detect_face(target_image_location)[0]
print("Target detected and aligned")
physical_image = detect_face(physical_image_location)[0]
print("physical face detected and aligned")
input_image.save('./results/example/combine-face.png')
target_image.save('./results/example/target-face.png')
physical_image.save('./results/example/physical-mask.png')

## Mask creation
mask = create_mask(input_image)[0]
cond = tensorize(mask).cuda()
delta = tensorize(mask).cuda()
delta.requires_grad_(True)

## Optimizer, some options to consider: Adam, SGD
opt = optim.Adamax([delta], lr = 1e-1, weight_decay = 0.0001)

## Initializing the FaceNet embeddings to be used in the loss function
input_emb = resnet(norm(tensorize(input_image).cuda()))
target_emb = resnet(norm(tensorize(target_image).cuda()))
print("Embeddings created")

## Will be used to combine with mask for training
input_tensor = tensorize(input_image).cuda()

## Number of training rounds
epochs = 40

## Adversarial training
## 'loss' maximizes the distance between the adversarial embedding and the
## original input embedding and minimizes the distance between the adversarial
## embedding and the target embedding
print(f'\nEpoch |   Loss   | Face Detection             | Difference ')
print(f'---------------------------------')
for i in range(epochs):

    adver = apply(cond, input_tensor, delta)
    adv = imagize(adver.detach().cpu())
    embedding = resnet(norm(adver))
    embedding1= resnet(norm(tensorize(adv).cuda()))

    loss = (-emb_distance(embedding, input_emb)
            +emb_distance(embedding, target_emb))

    ## Some pretty printing and testing to check whether face detection passes
    if i % 25 == 0 or i == epochs - 1:
        detection_test = fr.face_locations(np.array(adv))
        if not detection_test:
            d = 'Failed'
        else:
            d = 'Pass ' + str(detection_test)
        print(f'{i:5} | {loss.item():8.5f} | {d} | {emb_distance(resnet(norm(adver.cuda())), target_emb).item()}')
        print(emb_distance(resnet(norm(tensorize(adv).cuda())), target_emb).item())


        
        adv.show()

    ## Backprop step
    loss.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()

    delta.data.clamp_(0, 1)


imagize(delta.detach().cpu()).show()
adv.show()

## Additional testing image for the ground truth 
temp = detect_face(input_test_location)[0]
true_emb = resnet(norm(tensorize(temp).cuda()))
## Additional testing image for the target
temp = detect_face(target_test_location)[0]
test_emb = resnet(norm(tensorize(temp).cuda()))

## Additional attacker image for the target
temp = detect_face(input_test_location)[0]
attacker_tensor = tensorize(temp).cuda()


## Testing of physical mask
temp = detect_face(physical_image_location)[0]
physical_emb = resnet(norm(tensorize(temp).cuda()))

## Distance calculations and "pretty" printing
print("\nsame faces vs same faces")
print("target vs 2nd target  ", emb_distance(target_emb, test_emb).item())
print("input img vs true img  ", emb_distance(input_emb, true_emb).item())

print("\ninitial vs targets and advrs vs targets")
print("input img vs target    ", emb_distance(input_emb, target_emb).item())
print("input img vs 2nd target", emb_distance(input_emb, test_emb).item())


print("advrs img vs target    ", emb_distance(resnet(norm(apply(cond, input_tensor, delta))), target_emb).item())
print("advrs img vs 2nd target", emb_distance(resnet(norm(apply(cond, input_tensor, delta))), test_emb).item())

print("\nadvrs vs original inputs")
print("advrs img vs input img  ", emb_distance(resnet(norm(apply(cond, input_tensor, delta))), input_emb).item())
print("advrs img vs true img  ", emb_distance(resnet(norm(apply(cond, input_tensor, delta))), true_emb).item())

print("\nFor physical testing")
print("physical_mask vs target ", emb_distance(physical_emb, target_emb).item())
print("physical_mask vs 2nd target ", emb_distance(physical_emb, test_emb).item())
print("physical_mask vs advrs    ", emb_distance(physical_emb, resnet(norm(apply(cond, input_tensor, delta)))).item())
print("physical_mask vs adv    ", emb_distance(physical_emb, resnet(norm(tensorize(adv).cuda()))).item())

print("\nattacker vs target    ", emb_distance(resnet(norm(apply(cond, attacker_tensor, delta))), target_emb).item())
attacker = apply(cond, attacker_tensor, delta)
attacker_img = imagize(attacker.detach().cpu())
attacker_img.show()



## Final results
Image.fromarray(np.hstack(
    (np.asarray(input_image.resize((300,300))), 
     np.asarray(imagize(delta.detach().cpu()).resize((300,300))),
     np.asarray(imagize(apply(cond, input_tensor, delta).detach().cpu()).resize((300,300))),
     np.asarray(target_image.resize((300,300)))))).show()

if not(os.path.isdir('./results/example')):
    os.mkdir('./results/example')

imagize(delta.detach().cpu()).save('./results/example/delta.png')
imagize(apply(cond, input_tensor, delta).detach().cpu()).save('./results/example/combined-face.png')