from scipy.spatial import Voronoi, distance
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import numpy as np

# Points representing rubbish detection
points = np.array([
    [1, 2], [2, 3], [3, 1], [8, 8], [7, 9], [9, 7],
    [20, 20], [21, 21], [22, 20], [15, 2], [16, 3], [17, 1],
    [5, 18], [6, 19], [8, 17], [12, 12], [13, 13], [14, 14],
    [18, 10], [19, 11], [20, 9], [25, 25], [26, 26], [27, 27],
    [30, 15], [31, 16], [29, 14], [10, 4], [11, 20], [12, 4], [14, 3],
    [23, 2], [13, 10]
])

max_distance_threshold = 5.0

# Obstacles
obstacles = [
    # Boundaries
    Polygon([(0,0), (0.5,0), (0.5, 30), (0, 30)]), 
    Polygon([(0,30), (35,30), (35, 29.5), (0, 29.5)]),
    Polygon([(35,30), (35,0), (34.5, 0), (34.5, 30)]),
    Polygon([(0,0), (35,0), (35, 0.5), (0, 0.5)]),

    # Obstacles
    Polygon([(5, 5), (15, 5), (15, 6), (6, 6), (6, 10), (5, 10)]),  # L wall
    Polygon([(15, 15), (16, 15), (16, 16), (15, 16)]),  # another square wall
    Polygon([(25, 5), (26, 5), (26, 6), (25, 6)]),  # another square wall
    Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),  # small wall
    Polygon([(20, 25), (21, 25), (21, 26), (20, 26)])  # another small wall
]

# Check if a line segment intersects any obstacles
def intersects_obstacle(point1, point2, obstacles):
    line = LineString([point1, point2])
    return any(line.intersects(obstacle) for obstacle in obstacles)

# Initialize each point as its own cluster
clusters = [[p] for p in points]

# Constrained agglomerative clustering with distance threshold
while len(clusters) > 1:
    min_dist = np.inf
    closest_pair = None

    # Find closest pair of clusters
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for point1 in clusters[i]:
                for point2 in clusters[j]:
                    dist = distance.euclidean(point1, point2)

                    # Check that cluster is within max distance threshold for clustering
                    if dist < min_dist and dist <= max_distance_threshold and not intersects_obstacle(point1, point2, obstacles):
                        min_dist = dist
                        closest_pair = (i, j)

    # If no valid pair is found, exit
    if closest_pair is None:
        break

    # Merge the closest pair of clusters
    i, j = closest_pair
    clusters[i].extend(clusters[j])
    clusters.pop(j)

# Get cluster centroids by flattening
cluster_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

# Add boundary points to the centroids to ensure Voronoi region limits
boundary_points = np.array([
    [0, 0], [0, 30], [35, 30], [35, 0]
])
all_points = np.vstack([cluster_centroids, boundary_points])

# Compute Voronoi tessellation using the centroids and boundary points
vor = Voronoi(all_points)

# Define the boundary of the room (boundary walls)
# Required so that most region within the room would be assigned 
# a voronoi cluster
boundary_polygon = Polygon([
    (0, 0), (0, 30), (35, 30), (35, 0), (0, 0)
])

# Combine all obstacles into a single geometry for clipping
obstacles_union = unary_union(obstacles)

# Visualization of the Voronoi expansion
plt.figure(figsize=(10, 10))



## Plotting of results ##
# Plot Voronoi tessellation with boundary clipping and obstacle exclusion
for region_index in vor.regions:
    if not region_index or -1 in region_index:
        continue
    
    region = [vor.vertices[i] for i in region_index if i != -1]
    polygon = Polygon(region).intersection(boundary_polygon)  # Intersect with boundary polygon
    
    # Subtract obstacles from the Voronoi region
    polygon = polygon.difference(obstacles_union)
    
    # Handle MultiPolygon case (If we have a line segment)
    if isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.4)
    elif isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        plt.fill(x, y, alpha=0.4)

# Plot centroids
plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], color='red', s=100, marker='x', label='Centroids')

# Plot original points with cluster labels
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], color='blue', s=50, marker='o')
    for point in cluster:
        plt.text(point[0], point[1], f'{i+1}', fontsize=9, ha='center', va='center', color='white')

# Plot obstacles
for obstacle in obstacles:
    x, y = obstacle.exterior.xy
    plt.fill(x, y, color='gray', alpha=0.5)

# Plot boundary as a line
x, y = boundary_polygon.exterior.xy
plt.plot(x, y, color='black')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Voronoi Expansion Constrained by Boundary Walls and Obstacles')
plt.legend()
plt.grid(True)
plt.show()
