# Background-Removal plugin for ImageJ and ImageJ2 (Fiji) 
ImageJ is a fully java-based software for image processing in fields such as physics and biology. Our plugin aims to remove the background from a stack of images using the Low Rank + Sparse + Noise method from the article [Greedy Bilateral Sketch, Completion & Smoothing](https://proceedings.mlr.press/v31/zhou13b.html).
* Low Rank => Background matrix
* Sparse => Moving objects martrix
* Noise => Disruption and errors matrix

![](1-Original.gif)
