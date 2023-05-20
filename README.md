## SuperThermal-thermal-matching
An unofficial PyTorch implementation of SuperThermal: Matching Thermal as Visible Through Thermal Feature Exploration
This implementation is trained based on this paper and can get similar matching results, you can cite this work here:

```
@article{lu2021superthermal,
  title={SuperThermal: Matching thermal as visible through thermal feature exploration},
  author={Lu, Yawen and Lu, Guoyu},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={2690--2697},
  year={2021},
  publisher={IEEE}
}
```

## Usage for application, batch output matching results in a folder

*On kaist:
python batch_draw_matching_points.py --resume KAIST.pth.tar --folder test_input

*On CSS:
python batch_draw_matching_points.py --resume CSS.pth.tar --folder test_input
