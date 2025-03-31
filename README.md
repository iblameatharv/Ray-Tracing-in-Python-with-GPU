# GPU-Accelerated 2D Ray Tracing

A visual physics simulation demonstrating real-time ray tracing with reflection and refraction effects using PyGame and CUDA acceleration.

![Ray Tracing Demo](https://raw.githubusercontent.com/iblameatharv/gpu-ray-tracing/main/demo.gif)

## Features

- **Real-time 2D ray tracing** with up to 360 rays (GPU mode) or 180 rays (CPU mode)
- **Physics simulation** of light properties:
  - Reflection from mirror surfaces
  - Light dispersion through prism objects
  - Multiple ray bounces (3 in GPU mode, 2 in CPU mode)
- **CUDA GPU acceleration** for improved performance (with fallback to CPU)
- **Dynamic interactive environment** controlled by mouse movement
- **Visual effects** including rainbow dispersions, reflections, and ray visualization

## Requirements

- Python 3.6+
- PyGame
- NumPy
- Numba (for CUDA acceleration)
- NVIDIA GPU with CUDA support (optional, for acceleration)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/iblameatharv/gpu-ray-tracing.git
   cd gpu-ray-tracing
   ```

2. Install the required dependencies:
   ```
   pip install pygame numpy numba
   ```

3. For GPU acceleration (optional):
   - Install NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
   - Install additional Python packages:
     ```
     pip install cupy
     ```
   - Ensure your graphics drivers are up to date

## Usage

Run the simulation:
```
python ray_tracing.py
```

### Controls
- **Mouse movement**: Controls the position of the light source
- **ESC**: Exit the application

## How It Works

### Ray Tracing Algorithm
The simulation casts rays in all directions from a light source and calculates their intersection with various objects in the scene:

1. For each ray:
   - Calculate intersections with all boundaries (walls, mirrors, prisms)
   - Determine the closest intersection point
   - Render the ray path to that point

2. For special surfaces:
   - **Mirrors**: Calculate reflection angle and continue ray tracing
   - **Prisms**: Split the ray into multiple colored rays (dispersion effect)

### GPU Acceleration
When CUDA is available:
- Ray-wall intersection calculations are offloaded to the GPU
- Parallel processing allows for more rays and bounces
- The CUDA kernel handles multiple rays simultaneously

## Performance Notes

- **GPU Mode**: Targets 360 rays with 3 reflection bounces
- **CPU Mode**: Automatically reduces to 180 rays with 2 reflection bounces
- FPS counter displays current performance (Green: >150 FPS, Yellow: >100 FPS, Red: <100 FPS)

## Customization

You can modify these parameters in the code:
- `RAY_COUNT`: Number of rays to cast
- `REFLECTION_BOUNCES`: Maximum number of ray bounces
- Screen dimensions and colors
- Number and types of boundaries (walls, mirrors, prisms)

## Troubleshooting

If you encounter issues with CUDA:
1. Check if your GPU supports CUDA
2. Verify CUDA Toolkit installation
3. Update graphics drivers
4. The program will automatically fall back to CPU mode if CUDA fails

## License

[MIT License](LICENSE)

## Author

[iblameatharv](https://github.com/iblameatharv)

## Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a pull request
