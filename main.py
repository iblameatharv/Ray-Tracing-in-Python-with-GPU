import pygame
import sys
import math
import random
import numpy as np
import time
from pygame.locals import *

# Try to import CUDA, but provide instructions if it fails
try:
    from numba import jit, cuda
    CUDA_AVAILABLE = True
    print("CUDA successfully imported! Using GPU acceleration.")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA import failed! To enable GPU acceleration, follow these steps:")
    print("1. Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("2. Install Numba and CuPy: pip install numba cupy")
    print("3. Make sure your graphics drivers are up to date")
    print("Running in CPU-only mode with reduced ray count for performance.")

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GPU-Accelerated 2D Ray Tracing")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Define rainbow colors for prism effects
RAINBOW_COLORS = [
    (255, 0, 0),    # Red
    (255, 127, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (75, 0, 130),   # Indigo
    (148, 0, 211)   # Violet
]

# Set ray count based on CUDA availability
RAY_COUNT = 360 if CUDA_AVAILABLE else 180
REFLECTION_BOUNCES = 3 if CUDA_AVAILABLE else 2

# GPU-accelerated ray-wall intersection calculation
if CUDA_AVAILABLE:
    @cuda.jit
    def ray_wall_intersections(ray_pos_x, ray_pos_y, ray_dir_x, ray_dir_y, 
                              wall_x1, wall_y1, wall_x2, wall_y2, 
                              intersections, distances, wall_indices):
        i = cuda.grid(1)
        if i < ray_pos_x.shape[0]:
            # Ray parameters
            x3, y3 = ray_pos_x[i], ray_pos_y[i]
            x4, y4 = x3 + ray_dir_x[i], y3 + ray_dir_y[i]
            
            closest_dist = 1e9  # Large number
            closest_point_x = 1e9
            closest_point_y = 1e9
            closest_wall_idx = -1
            
            for j in range(wall_x1.shape[0]):
                # Wall parameters
                x1, y1 = wall_x1[j], wall_y1[j]
                x2, y2 = wall_x2[j], wall_y2[j]
                
                # Calculate denominator
                den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                
                # Lines are parallel
                if abs(den) < 1e-6:
                    continue
                
                # Calculate parameters t and u
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
                
                # Check if intersection is valid
                if 0 <= t <= 1 and u > 0:
                    # Calculate intersection point
                    point_x = x1 + t * (x2 - x1)
                    point_y = y1 + t * (y2 - y1)
                    
                    # Calculate distance
                    dist = math.sqrt((x3 - point_x)**2 + (y3 - point_y)**2)
                    
                    # Update if this is the closest intersection
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point_x = point_x
                        closest_point_y = point_y
                        closest_wall_idx = j
            
            # Store the closest intersection
            if closest_wall_idx >= 0:
                intersections[i, 0] = closest_point_x
                intersections[i, 1] = closest_point_y
                distances[i] = closest_dist
                wall_indices[i] = closest_wall_idx

# Ray class
class Ray:
    def __init__(self, pos, angle, color=WHITE):
        self.pos = pos
        self.dir = (math.cos(angle), math.sin(angle))
        self.color = color
    
    def show(self):
        pygame.draw.line(screen, self.color, self.pos, 
                        (self.pos[0] + self.dir[0] * 10, 
                         self.pos[1] + self.dir[1] * 10), 1)
    
    def cast(self, wall):
        # Line segment parameters
        x1, y1 = wall.a
        x2, y2 = wall.b
        
        # Ray parameters
        x3, y3 = self.pos
        x4, y4 = self.pos[0] + self.dir[0], self.pos[1] + self.dir[1]
        
        # Calculate denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Lines are parallel
        if abs(den) < 1e-6:
            return None
        
        # Calculate parameters t and u
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        
        # Check if intersection is valid
        if 0 <= t <= 1 and u > 0:
            # Calculate intersection point
            point = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            return point
        
        return None

# Boundary class (wall, mirror or prism)
class Boundary:
    def __init__(self, x1, y1, x2, y2, boundary_type="wall"):
        self.a = (x1, y1)
        self.b = (x2, y2)
        self.type = boundary_type  # "wall", "mirror", or "prism"
        
        # Calculate normal vector for reflection and refraction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            self.normal = (-dy/length, dx/length)  # Perpendicular to the wall
        else:
            self.normal = (0, 1)  # Default if length is 0
    
    def show(self):
        if self.type == "wall":
            pygame.draw.line(screen, WHITE, self.a, self.b, 2)
        elif self.type == "mirror":
            pygame.draw.line(screen, (200, 200, 255), self.a, self.b, 3)
        elif self.type == "prism":
            # Draw with rainbow gradient
            for i in range(len(RAINBOW_COLORS)):
                t1 = i / len(RAINBOW_COLORS)
                t2 = (i + 1) / len(RAINBOW_COLORS)
                x1 = self.a[0] + (self.b[0] - self.a[0]) * t1
                y1 = self.a[1] + (self.b[1] - self.a[1]) * t1
                x2 = self.a[0] + (self.b[0] - self.a[0]) * t2
                y2 = self.a[1] + (self.b[1] - self.a[1]) * t2
                pygame.draw.line(screen, RAINBOW_COLORS[i], (x1, y1), (x2, y2), 5)

# Particle class (light source)
class Particle:
    def __init__(self, x, y):
        self.pos = (x, y)
        self.rays = []
        self.heading = 0
        self.max_bounces = REFLECTION_BOUNCES
        self.create_rays(RAY_COUNT)
        
        # For GPU acceleration
        self.update_arrays()
    
    def create_rays(self, count):
        self.rays = []
        for i in range(count):
            angle = math.radians(i * (360 / count))
            self.rays.append(Ray(self.pos, angle))
    
    def update(self, x, y):
        self.pos = (x, y)
        for ray in self.rays:
            ray.pos = self.pos
        self.update_arrays()
    
    def update_arrays(self):
        if CUDA_AVAILABLE:
            self.ray_pos_x = np.array([self.pos[0]] * len(self.rays), dtype=np.float32)
            self.ray_pos_y = np.array([self.pos[1]] * len(self.rays), dtype=np.float32)
            self.ray_dir_x = np.array([ray.dir[0] for ray in self.rays], dtype=np.float32)
            self.ray_dir_y = np.array([ray.dir[1] for ray in self.rays], dtype=np.float32)
    
    def look(self, walls):
        if CUDA_AVAILABLE:
            self.look_gpu(walls)
        else:
            self.look_cpu(walls)
    
    def look_cpu(self, walls):
        # CPU fallback implementation
        for ray in self.rays:
            closest = None
            record = float('inf')
            closest_wall = None
            
            for wall in walls:
                point = ray.cast(wall)
                if point:
                    # Calculate distance
                    d = math.sqrt((self.pos[0] - point[0])**2 + (self.pos[1] - point[1])**2)
                    
                    if d < record:
                        record = d
                        closest = point
                        closest_wall = wall
            
            if closest:
                # Draw a line to the closest intersection
                pygame.draw.line(screen, ray.color, self.pos, closest, 1)
                
                # Handle reflections and refractions
                if closest_wall.type == "mirror" and self.max_bounces > 0:
                    # Calculate reflection
                    incoming = ray.dir
                    normal = closest_wall.normal
                    
                    # r = d - 2(d·n)n
                    dot = incoming[0]*normal[0] + incoming[1]*normal[1]
                    reflected_dir = (
                        incoming[0] - 2 * dot * normal[0],
                        incoming[1] - 2 * dot * normal[1]
                    )
                    
                    # Create reflection ray
                    reflection = Ray(closest, 0, ray.color)
                    reflection.dir = reflected_dir
                    
                    # Recursive reflection (limited depth)
                    self.trace_ray(reflection, walls, self.max_bounces - 1)
                
                elif closest_wall.type == "prism" and self.max_bounces > 0:
                    # Create dispersed rays (simplified prism effect)
                    for j, color in enumerate(RAINBOW_COLORS):
                        # Create a slightly different angle for each color
                        angle_offset = (j - len(RAINBOW_COLORS)/2) * 0.1
                        base_angle = math.atan2(ray.dir[1], ray.dir[0])
                        new_angle = base_angle + angle_offset
                        
                        # Create dispersed ray
                        dispersed = Ray(closest, new_angle, color)
                        
                        # Trace the dispersed ray
                        self.trace_ray(dispersed, walls, self.max_bounces - 1)
    
    def trace_ray(self, ray, walls, bounces_left):
        if bounces_left <= 0:
            return
            
        closest = None
        record = float('inf')
        closest_wall = None
        
        for wall in walls:
            point = ray.cast(wall)
            if point:
                # Calculate distance
                d = math.sqrt((ray.pos[0] - point[0])**2 + (ray.pos[1] - point[1])**2)
                
                if d < record:
                    record = d
                    closest = point
                    closest_wall = wall
        
        if closest:
            # Draw the ray
            pygame.draw.line(screen, ray.color, ray.pos, closest, 1)
            
            # Handle reflections and refractions
            if closest_wall.type == "mirror" and bounces_left > 0:
                # Calculate reflection
                incoming = ray.dir
                normal = closest_wall.normal
                
                # r = d - 2(d·n)n
                dot = incoming[0]*normal[0] + incoming[1]*normal[1]
                reflected_dir = (
                    incoming[0] - 2 * dot * normal[0],
                    incoming[1] - 2 * dot * normal[1]
                )
                
                # Create reflection ray
                reflection = Ray(closest, 0, ray.color)
                reflection.dir = reflected_dir
                
                # Recursive reflection
                self.trace_ray(reflection, walls, bounces_left - 1)
            
            elif closest_wall.type == "prism" and bounces_left > 0:
                # Create dispersed rays (simplified prism effect)
                for j, color in enumerate(RAINBOW_COLORS):
                    # Create a slightly different angle for each color
                    angle_offset = (j - len(RAINBOW_COLORS)/2) * 0.1
                    base_angle = math.atan2(ray.dir[1], ray.dir[0])
                    new_angle = base_angle + angle_offset
                    
                    # Create dispersed ray
                    dispersed = Ray(closest, new_angle, color)
                    
                    # Trace the dispersed ray
                    self.trace_ray(dispersed, walls, bounces_left - 1)
    
    def look_gpu(self, walls):
        if not CUDA_AVAILABLE or len(walls) == 0 or len(self.rays) == 0:
            self.look_cpu(walls)
            return
        
        # Prepare wall data
        wall_count = len(walls)
        wall_x1 = np.zeros(wall_count, dtype=np.float32)
        wall_y1 = np.zeros(wall_count, dtype=np.float32)
        wall_x2 = np.zeros(wall_count, dtype=np.float32)
        wall_y2 = np.zeros(wall_count, dtype=np.float32)
        wall_types = []
        
        for i, wall in enumerate(walls):
            wall_x1[i] = wall.a[0]
            wall_y1[i] = wall.a[1]
            wall_x2[i] = wall.b[0]
            wall_y2[i] = wall.b[1]
            wall_types.append(wall.type)
        
        # Recursive ray tracing for reflections and refractions
        self.trace_rays_gpu(walls, wall_x1, wall_y1, wall_x2, wall_y2, wall_types, 0)
    
    def trace_rays_gpu(self, walls, wall_x1, wall_y1, wall_x2, wall_y2, wall_types, bounce_count):
        if bounce_count >= self.max_bounces:
            return
        
        ray_count = len(self.rays)
        
        # Prepare arrays for GPU
        intersections = np.full((ray_count, 2), float('inf'), dtype=np.float32)
        distances = np.full(ray_count, float('inf'), dtype=np.float32)
        wall_indices = np.full(ray_count, -1, dtype=np.int32)
        
        # Configure the blocks and grid for CUDA
        threadsperblock = 128
        blockspergrid = (ray_count + (threadsperblock - 1)) // threadsperblock
        
        try:
            # Run the CUDA kernel
            ray_wall_intersections[blockspergrid, threadsperblock](
                self.ray_pos_x, self.ray_pos_y, self.ray_dir_x, self.ray_dir_y,
                wall_x1, wall_y1, wall_x2, wall_y2,
                intersections, distances, wall_indices
            )
            
            # Process results and handle reflections/refractions
            new_rays = []
            
            for i in range(ray_count):
                if wall_indices[i] >= 0:
                    point = (intersections[i, 0], intersections[i, 1])
                    wall_idx = int(wall_indices[i])
                    
                    # Draw ray to intersection
                    pygame.draw.line(screen, self.rays[i].color, 
                                    (self.ray_pos_x[i], self.ray_pos_y[i]), 
                                    point, 1)
                    
                    wall_type = wall_types[wall_idx]
                    if wall_type == "mirror":
                        # Calculate reflection
                        wall = walls[wall_idx]
                        incoming = (self.ray_dir_x[i], self.ray_dir_y[i])
                        normal = wall.normal
                        
                        # r = d - 2(d·n)n
                        dot = incoming[0]*normal[0] + incoming[1]*normal[1]
                        reflected_dir = (
                            incoming[0] - 2 * dot * normal[0],
                            incoming[1] - 2 * dot * normal[1]
                        )
                        
                        # Add a new ray for reflection
                        new_ray = Ray(point, 0, self.rays[i].color)
                        new_ray.dir = reflected_dir
                        new_rays.append(new_ray)
                        
                    elif wall_type == "prism":
                        # Create dispersed rays
                        for j, color in enumerate(RAINBOW_COLORS):
                            angle_offset = (j - len(RAINBOW_COLORS)/2) * 0.1
                            base_angle = math.atan2(self.ray_dir_y[i], self.ray_dir_x[i])
                            new_angle = base_angle + angle_offset
                            
                            new_ray = Ray(point, new_angle, color)
                            new_rays.append(new_ray)
            
            # Continue tracing for new rays
            if new_rays and bounce_count < self.max_bounces:
                self.rays = new_rays
                self.update_arrays()
                self.trace_rays_gpu(walls, wall_x1, wall_y1, wall_x2, wall_y2, wall_types, bounce_count + 1)
                
        except Exception as e:
            print(f"GPU acceleration failed: {e}")
            print("Falling back to CPU implementation")
            self.look_cpu(walls)
    
    def show(self):
        pygame.draw.circle(screen, WHITE, (int(self.pos[0]), int(self.pos[1])), 4)

# Create random walls, mirrors, and prisms
def create_boundaries():
    boundaries = []
    
    # Add border walls
    boundaries.append(Boundary(0, 0, WIDTH, 0, "wall"))
    boundaries.append(Boundary(WIDTH, 0, WIDTH, HEIGHT, "wall"))
    boundaries.append(Boundary(WIDTH, HEIGHT, 0, HEIGHT, "wall"))
    boundaries.append(Boundary(0, HEIGHT, 0, 0, "wall"))
    
    # Add random mirrors (5)
    for _ in range(5):
        x1 = random.randint(50, WIDTH - 50)
        y1 = random.randint(50, HEIGHT - 50)
        x2 = x1 + random.randint(-100, 100)
        y2 = y1 + random.randint(-100, 100)
        boundaries.append(Boundary(x1, y1, x2, y2, "mirror"))
    
    # Add random prisms (3)
    for _ in range(3):
        x1 = random.randint(50, WIDTH - 50)
        y1 = random.randint(50, HEIGHT - 50)
        x2 = x1 + random.randint(-80, 80)
        y2 = y1 + random.randint(-80, 80)
        boundaries.append(Boundary(x1, y1, x2, y2, "prism"))
    
    # Add some random walls (7)
    for _ in range(7):
        x1 = random.randint(50, WIDTH - 50)
        y1 = random.randint(50, HEIGHT - 50)
        x2 = x1 + random.randint(-150, 150)
        y2 = y1 + random.randint(-150, 150)
        boundaries.append(Boundary(x1, y1, x2, y2, "wall"))
    
    return boundaries

# Performance monitoring
def show_fps(clock, font):
    fps = int(clock.get_fps())
    fps_text = font.render(f"FPS: {fps}", True, GREEN if fps > 150 else YELLOW if fps > 100 else RED)
    screen.blit(fps_text, (10, 10))
    
    # Show acceleration mode
    mode_text = font.render(f"Mode: {'GPU' if CUDA_AVAILABLE else 'CPU'}", True, GREEN if CUDA_AVAILABLE else YELLOW)
    screen.blit(mode_text, (10, 30))
    
    # Show controls
    controls_text = font.render("Controls: Mouse to move light source", True, WHITE)
    screen.blit(controls_text, (10, 50))

# Main function
def main():
    clock = pygame.time.Clock()
    
    # Font for FPS display
    font = pygame.font.SysFont('Arial', 18)
    
    # Create boundaries with random mirrors and prisms
    boundaries = create_boundaries()
    
    # Create light source at center
    particle = Particle(WIDTH // 2, HEIGHT // 2)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
        
        # Get mouse position to control the light source
        mouse_x, mouse_y = pygame.mouse.get_pos()
        particle.update(mouse_x, mouse_y)
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw boundaries
        for boundary in boundaries:
            boundary.show()
        
        # Draw rays and intersections
        particle.look(boundaries)
        particle.show()
        
        # Show FPS counter and mode
        show_fps(clock, font)
        
        pygame.display.flip()
        
        # Run at maximum possible FPS
        clock.tick(0)  # 0 means unlimited FPS
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
