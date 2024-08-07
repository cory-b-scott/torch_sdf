# torch_sdf

![Optimization of a set of 2D spherical SDFs to match a set image.](https://github.com/cory-b-scott/img_assets/blob/main/merged.gif)

This is a project intended to facilitate manipulation of, and learning through, signed distance functions.

It consists of operations for:

- Making SDFs that represent basic geometric primitives in 2D, 3D (and in some cases nD);
- Applying affine transformations to SDFs;
- Combining primitives to make complex shapes (union, intersection, XOR, difference);
- More complicated manipulation of SDFs to produce other SDFs (dilation, erosion, onioning, extension, inversion);
- Building Neural SDFs;
- Repairing "broken" SDFS;

All of the above operations are implemented both object-oriented flavor (AKA torch.nn.Modules) and in a functional form (like torch.nn.functional).

Any operation can take either a torch.Tensor or a torch.nn.Parameter for any of its inputs. With some exceptions[^1] this means that it is possible to optimize either the parameters of an SDF w.r.t. a set of query points or vice versa - see Examples below.

This package could not exist without the extensive work on SDFs by [Inigo Quilez](https://iquilezles.org/). Especially his library of [2D](https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm) and [3D SDFs](https://iquilezles.org/www/articles/distfunctions/distfunctions.htm), as well as his articles on the [smooth-min operation](https://iquilezles.org/articles/smin/) and [bounding boxes](https://iquilezles.org/articles/sdfbounding/). Deep thanks to Inigo for this extensive and thorough work.

A related package which predates this one is [sdf](https://github.com/fogleman/sdf), which is a similar idea but with a numpy backend and efficient parallelization.

[^1]: mostly cases where SDFs are composed of non-differentiable functions and there's no way around that.
## Examples
TODO

## Requirements
- torch (2.*)
