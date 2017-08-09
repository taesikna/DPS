

im = imread('../../data/ilsvrc12/train/n01440764/n01440764_10026.JPEG');
%im = imread('../../examples/images/cat.jpg');
scores = classification_demo(im, 1);
[score, class] = max(scores)

