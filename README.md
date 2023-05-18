# Background-Removal plugin for ImageJ and ImageJ2 (Fiji) 
ImageJ is a fully java-based software for image processing in fields such as physics and biology. Our plugin aims to remove the background from a stack of images using the Low Rank + Sparse + Noise method from the article [Greedy Bilateral Sketch, Completion & Smoothing](https://proceedings.mlr.press/v31/zhou13b.html).
* Low Rank => Background matrix
* Sparse => Moving objects martrix
* Noise => Disruption and errors matrix

## Example of Low Rank + Sparse + Noise method
**Original images:** | **Background images:** | **Sparse images:** | **Noise:**
--- | --- | --- | ---
![Original](/samples/gifs/1-Original.gif) | ![Background](/samples/gifs/2-Background.gif) | ![Sparse](/samples/gifs/3-Sparse.gif) | ![Noise](/samples/gifs/4-Noise_1.gif)

## Installation tutorial
###### Requirement:  [ImageJ (Fiji)](https://imagej.net/software/fiji/downloads)
You can get the `.jar` of this plugin from the plugins folder [here](https://sites.imagej.net/FattaccioliLab/). 
Or from the project folder with Maven :
```bash
cd  Background-Removal-Plugin---ImageJ
```
```bash
mvn clean package -Denforcer.skip=true
```
```bash
cd  target
```
The file is named `BackgroundRemoval-0.1.0-SNAPSHOT.jar` in the project folder and `GreGoDec-v0.1.jar` in the ImageJ website. After you get the .jar plugin, you just need to paste it into plugins Fiji.app folder.

## How to use it ?
You should now see `Background Removal` in the Fiji `Plugins` bar:
![Start](/samples/gifs/ImageJ_1.png)
After clicking, the GUI appears:
![Interface](/samples/gifs/interface_1.png)