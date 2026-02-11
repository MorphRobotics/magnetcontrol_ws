%% ===================================================================
%% PART 1: CREATE HIGH-RESOLUTION OBSTACLE MAP (0.5mm voxels)
%% ===================================================================

clc; clear; close all;

stlFile = 'Bronchial tree anatomy-1mm-shell (1).STL';
voxelSize = 0.5; % HIGH RESOLUTION (was 1.0mm)

fprintf('====================================\n');
fprintf('Creating HIGH-RES obstacle map\n');
fprintf('Voxel size: %.2f mm\n', voxelSize);
fprintf('====================================\n\n');

%% Read STL
fprintf('Reading STL...\n');
TR = stlread(stlFile);
vertices = TR.Points;
faces = TR.ConnectivityList;
fprintf('  %d vertices, %d faces\n', size(vertices,1), size(faces,1));

%% Create high-res grid
minBounds = min(vertices, [], 1);
maxBounds = max(vertices, [], 1);
dimensions = maxBounds - minBounds;
gridSize = ceil(dimensions / voxelSize) + 1;

fprintf('\nGrid: %d × %d × %d voxels\n', gridSize);
fprintf('Memory: %.1f MB\n', prod(gridSize) / 1024^2 / 8);

%% Voxelize with inpolyhedron
fprintf('\nVoxelizing (this may take 2-3 minutes for high-res)...\n');
tic;

x = linspace(minBounds(1), maxBounds(1), gridSize(1));
y = linspace(minBounds(2), maxBounds(2), gridSize(2));
z = linspace(minBounds(3), maxBounds(3), gridSize(3));

[X, Y, Z] = meshgrid(x, y, z);
points = [X(:), Y(:), Z(:)];

in = inpolyhedron(faces, vertices, points);

obstacleMap = reshape(in, gridSize(2), gridSize(1), gridSize(3));
obstacleMap = permute(obstacleMap, [2, 1, 3]);
obstacleMap = logical(obstacleMap);

elapsed = toc;
fprintf('✓ Voxelization complete in %.1f seconds\n', elapsed);

%% Store grid info
gridInfo.voxelSize = voxelSize;
gridInfo.gridSize = gridSize;
gridInfo.minBounds = minBounds;
gridInfo.maxBounds = maxBounds;
gridInfo.dimensions = dimensions;
gridInfo.x = x;
gridInfo.y = y;
gridInfo.z = z;

save('bronchial_obstacle_map_highres.mat', 'obstacleMap', 'gridInfo');
fprintf('✓ Saved bronchial_obstacle_map_highres.mat\n\n');


%% ===================================================================
%% PART 2: BUILD ROADMAP WITH HIGH-RES MAP
%% ===================================================================

fprintf('Building roadmap on high-res map...\n');

load('prm_roadmap.mat', 'nodesFiltered', 'edgesFiltered');
nodes = nodesFiltered;
edges = edgesFiltered;

fprintf('Loaded roadmap: %d nodes, %d edges\n\n', size(nodes,1), size(edges,1));


%% ===================================================================
%% PART 3: STUNNING 3D VISUALIZATION
%% ===================================================================

fprintf('Creating 3D visualization...\n');

figure('Position', [100 100 1200 900], 'Color', 'w', 'Name', '3D Lung Roadmap');
hold on;

%% Show airway structure (semi-transparent)
fprintf('  Rendering airway surface...\n');
airwayInterior = obstacleMap;
[faces3d, verts3d] = isosurface(airwayInterior, 0.5);

if ~isempty(faces3d)
    % Scale to world coordinates
    verts3d(:,1) = gridInfo.x(1) + (verts3d(:,1)-1) * gridInfo.voxelSize;
    verts3d(:,2) = gridInfo.y(1) + (verts3d(:,2)-1) * gridInfo.voxelSize;
    verts3d(:,3) = gridInfo.z(1) + (verts3d(:,3)-1) * gridInfo.voxelSize;
    
    % Smooth rendering with reduced faces
    patch('Faces', faces3d, 'Vertices', verts3d, ...
          'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.25, 'FaceLighting', 'gouraud');
end

%% Draw pathway edges (green)
fprintf('  Drawing %d pathway edges...\n', size(edges,1));
for e = 1:size(edges,1)
    plot3([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
          [nodes(edges(e,1),2), nodes(edges(e,2),2)], ...
          [nodes(edges(e,1),3), nodes(edges(e,2),3)], ...
          '-', 'LineWidth', 2, 'Color', [0.2 0.8 0.2]);
end

%% Draw nodes (red spheres)
fprintf('  Drawing %d nodes...\n', size(nodes,1));
scatter3(nodes(:,1), nodes(:,2), nodes(:,3), 50, [0.9 0.1 0.1], 'filled', ...
         'MarkerEdgeColor', 'k', 'LineWidth', 0.8, 'MarkerFaceAlpha', 0.9);

%% Formatting
xlabel('\textbf{X (mm)}', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('\textbf{Y (mm)}', 'FontSize', 14, 'Interpreter', 'latex');
zlabel('\textbf{Z (mm)}', 'FontSize', 14, 'Interpreter', 'latex');

title(sprintf('\\textbf{3D Lung Navigation Roadmap: %d Nodes, %d Pathways}', ...
    size(nodes,1), size(edges,1)), 'FontSize', 16, 'Interpreter', 'latex');

axis equal;
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.5, 'GridAlpha', 0.3);

% Multiple light sources for better depth perception
camlight('headlight');
camlight('right');
lighting gouraud;

% Set good viewing angle
view(45, 25);

% Enable rotation
rotate3d on;

hold off;

print('roadmap_3d', '-dpng', '-r300');
fprintf('✓ Saved roadmap_3d.png\n');


%% ===================================================================
%% BONUS: ANIMATED 360° ROTATION
%% ===================================================================

fprintf('\nCreating 360° rotation animation...\n');

figure('Position', [100 100 1200 900], 'Color', 'w');
hold on;

% Render scene
if ~isempty(faces3d)
    patch('Faces', faces3d, 'Vertices', verts3d, ...
          'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.3, 'FaceLighting', 'gouraud');
end

for e = 1:size(edges,1)
    plot3([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
          [nodes(edges(e,1),2), nodes(edges(e,2),2)], ...
          [nodes(edges(e,1),3), nodes(edges(e,2),3)], ...
          '-', 'LineWidth', 1.5, 'Color', [0.2 0.8 0.2]);
end

scatter3(nodes(:,1), nodes(:,2), nodes(:,3), 40, [0.9 0.1 0.1], 'filled', ...
         'MarkerEdgeColor', 'k', 'LineWidth', 0.6);

xlabel('X (mm)', 'FontSize', 12); 
ylabel('Y (mm)', 'FontSize', 12); 
zlabel('Z (mm)', 'FontSize', 12);
title('3D Lung Roadmap - Rotating View', 'FontSize', 14, 'FontWeight', 'bold');

axis equal; grid on;
camlight('headlight'); camlight('right'); lighting gouraud;

hold off;

% Animate rotation
fprintf('  Rotating through 360°...\n');
for angle = 0:5:360
    view(angle, 25);
    drawnow;
    pause(0.05);
    
    if mod(angle, 90) == 0
        fprintf('    %d°\n', angle);
    end
end

fprintf('✓ Animation complete\n');


%% ===================================================================
%% INTERACTIVE 3D VIEW WITH CONTROLS
%% ===================================================================

fprintf('\nCreating interactive view...\n');

fig = figure('Position', [100 100 1400 1000], 'Color', 'w', ...
             'Name', 'Interactive 3D Roadmap - Use mouse to rotate');

% Main 3D view
ax1 = subplot(2,2,[1,3]);
hold on;

if ~isempty(faces3d)
    patch('Faces', faces3d, 'Vertices', verts3d, ...
          'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none', ...
          'FaceAlpha', 0.25, 'FaceLighting', 'gouraud');
end

for e = 1:size(edges,1)
    plot3([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
          [nodes(edges(e,1),2), nodes(edges(e,2),2)], ...
          [nodes(edges(e,1),3), nodes(edges(e,2),3)], ...
          '-', 'LineWidth', 2, 'Color', [0.2 0.8 0.2]);
end

scatter3(nodes(:,1), nodes(:,2), nodes(:,3), 50, [0.9 0.1 0.1], 'filled');

xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('3D Interactive View - Click and drag to rotate', 'FontSize', 14, 'FontWeight', 'bold');
axis equal; grid on;
camlight; lighting gouraud;
view(45, 25);
rotate3d on;

% XY view
subplot(2,2,2);
sliceZ = round(gridInfo.gridSize(3)/2);
slice = airwayInterior(:, :, sliceZ);
imagesc(gridInfo.x, gridInfo.y, slice');
colormap(gca, flipud(gray));
hold on;
for e = 1:size(edges,1)
    plot([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
         [nodes(edges(e,1),2), nodes(edges(e,2),2)], 'g-', 'LineWidth', 1);
end
scatter(nodes(:,1), nodes(:,2), 20, 'r', 'filled');
title('XY View (Top)', 'FontWeight', 'bold');
xlabel('X (mm)'); ylabel('Y (mm)');
axis equal tight; set(gca, 'YDir', 'normal');

% XZ view
subplot(2,2,4);
sliceY = round(gridInfo.gridSize(2)/2);
slice = squeeze(airwayInterior(:, sliceY, :));
imagesc(gridInfo.x, gridInfo.z, slice');
colormap(gca, flipud(gray));
hold on;
for e = 1:size(edges,1)
    plot([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
         [nodes(edges(e,1),3), nodes(edges(e,2),3)], 'g-', 'LineWidth', 1);
end
scatter(nodes(:,1), nodes(:,3), 20, 'r', 'filled');
title('XZ View (Side)', 'FontWeight', 'bold');
xlabel('X (mm)'); ylabel('Z (mm)');
axis equal tight; set(gca, 'YDir', 'normal');

print('roadmap_3d_interactive', '-dpng', '-r300');

fprintf('\n====================================\n');
fprintf('✓ ALL VISUALIZATIONS COMPLETE!\n');
fprintf('====================================\n');
fprintf('High-res map: bronchial_obstacle_map_highres.mat (%.2fmm voxels)\n', voxelSize);
fprintf('3D views saved:\n');
fprintf('  - roadmap_3d.png (main 3D view)\n');
fprintf('  - roadmap_3d_interactive.png (multi-view)\n');
fprintf('\nThe 3D figure is interactive - rotate it with your mouse!\n');