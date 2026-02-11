%% ===================================================================
%% DENSE K-NN ROADMAP - COVER ENTIRE LUNG
%% ===================================================================

clc; clear; close all;

load('bronchial_obstacle_map.mat', 'obstacleMap', 'gridInfo');
airwayInterior = obstacleMap;

%% DENSE SAMPLING PARAMETERS
config.numNodes = 5000;  % MUCH MORE NODES for full coverage
config.k = 25;           % Connect to 10 nearest neighbors
config.maxConnectionDist = 100;  % mm

fprintf('====================================\n');
fprintf('DENSE Roadmap: %d nodes, k=%d\n', config.numNodes, config.k);
fprintf('====================================\n\n');

%% Dense sampling across entire lung
fprintf('Sampling %d nodes throughout lung...\n', config.numNodes);
nodes = sampleDense(airwayInterior, gridInfo, config.numNodes);

%% Build k-NN graph
fprintf('\nBuilding k-NN graph (k=%d)...\n', config.k);
edges = buildKNN_Edges(nodes, airwayInterior, gridInfo, config.k, config.maxConnectionDist);

%% Analyze
adjList = cell(size(nodes,1), 1);
for e = 1:size(edges,1)
    adjList{edges(e,1)} = [adjList{edges(e,1)}, edges(e,2)];
    adjList{edges(e,2)} = [adjList{edges(e,2)}, edges(e,1)];
end
degrees = cellfun(@length, adjList);

fprintf('\n=== ROADMAP STATISTICS ===\n');
fprintf('Nodes: %d\n', size(nodes,1));
fprintf('Edges: %d\n', size(edges,1));
fprintf('Avg degree: %.2f\n', mean(degrees));
fprintf('Min/Max degree: %d/%d\n', min(degrees), max(degrees));

%% Find connected components
fprintf('\nAnalyzing connectivity...\n');
[components, sizes] = findConnectedComponents(size(nodes,1), adjList);
fprintf('Connected components: %d\n', length(sizes));

% Show all major components
[sortedSizes, sortedIdx] = sort(sizes, 'descend');
for i = 1:min(5, length(sortedSizes))
    fprintf('  Component %d: %d nodes (%.1f%%)\n', i, sortedSizes(i), ...
        100*sortedSizes(i)/size(nodes,1));
end

%% Keep ALL major components (>50 nodes)
majorComponents = find(sizes > 50);
fprintf('\nKeeping %d major components (>50 nodes each)\n', length(majorComponents));

keepNodes = ismember(components, majorComponents);
nodesFiltered = nodes(keepNodes, :);

% Remap edges
nodeMap = zeros(size(nodes,1), 1);
nodeMap(keepNodes) = 1:sum(keepNodes);

edgesFiltered = [];
for e = 1:size(edges,1)
    if nodeMap(edges(e,1)) > 0 && nodeMap(edges(e,2)) > 0
        edgesFiltered = [edgesFiltered; nodeMap(edges(e,1)), nodeMap(edges(e,2))];
    end
end

fprintf('Final roadmap: %d nodes, %d edges\n', size(nodesFiltered,1), size(edgesFiltered,1));

% Save
save('prm_roadmap.mat', 'nodesFiltered', 'edgesFiltered', 'config', 'gridInfo', 'airwayInterior');

%% Visualize
visualizeDense(nodesFiltered, edgesFiltered, airwayInterior, gridInfo);


%% ===================================================================
%% DENSE SAMPLING - COVER ALL REGIONS
%% ===================================================================

function nodes = sampleDense(airwayInterior, gridInfo, numNodes)
    % Sample nodes uniformly throughout airway volume
    
    [iAir, jAir, kAir] = ind2sub(size(airwayInterior), find(airwayInterior));
    
    minX = min(gridInfo.x(iAir)); maxX = max(gridInfo.x(iAir));
    minY = min(gridInfo.y(jAir)); maxY = max(gridInfo.y(jAir));
    minZ = min(gridInfo.z(kAir)); maxZ = max(gridInfo.z(kAir));
    
    fprintf('  Airway volume: [%.0f-%.0f] x [%.0f-%.0f] x [%.0f-%.0f] mm\n', ...
        minX, maxX, minY, maxY, minZ, maxZ);
    
    nodes = zeros(numNodes, 3);
    count = 0;
    maxAttempts = numNodes * 2000; % More attempts for dense sampling
    
    for attempt = 1:maxAttempts
        if count >= numNodes, break; end
        
        % Random point in airway bounds
        point = [minX + rand()*(maxX-minX), ...
                 minY + rand()*(maxY-minY), ...
                 minZ + rand()*(maxZ-minZ)];
        
        % Check if in airway
        [i,j,k,v] = world2grid(point, gridInfo);
        if v && airwayInterior(i,j,k)
            count = count + 1;
            nodes(count,:) = point;
            
            if mod(count,200)==0
                fprintf('    %d nodes sampled...\n', count);
            end
        end
    end
    
    if count < numNodes
        warning('Only sampled %d/%d nodes after %d attempts', count, numNodes, maxAttempts);
        nodes = nodes(1:count,:);
    else
        fprintf('  ✓ Sampled %d nodes\n', count);
    end
end


%% ===================================================================
%% K-NEAREST NEIGHBOR CONNECTION
%% ===================================================================

function edges = buildKNN_Edges(nodes, airwayInterior, gridInfo, k, maxDist)
    numNodes = size(nodes,1);
    edgeSet = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    
    fprintf('  Building k-NN graph...\n');
    
    for i = 1:numNodes
        % Compute distances to all other nodes
        distances = sqrt(sum((nodes - nodes(i,:)).^2, 2));
        [sortedDist, sortedIdx] = sort(distances);
        
        % Try to connect to k nearest neighbors
        connected = 0;
        for j = 2:min(k+20, length(sortedIdx)) % Try more than k
            if connected >= k, break; end
            
            neighbor = sortedIdx(j);
            dist = sortedDist(j);
            
            if dist > maxDist, break; end
            
            % Check if path is collision-free
            if isPathClear(nodes(i,:), nodes(neighbor,:), airwayInterior, gridInfo)
                key = sprintf('%d-%d', min(i,neighbor), max(i,neighbor));
                if ~isKey(edgeSet, key)
                    edgeSet(key) = true;
                    connected = connected + 1;
                end
            end
        end
        
        if mod(i,100)==0
            fprintf('    %d/%d nodes (%d edges so far)\n', i, numNodes, edgeSet.Count);
        end
    end
    
    % Convert to array
    edgeKeys = keys(edgeSet);
    edges = zeros(length(edgeKeys), 2);
    for e = 1:length(edgeKeys)
        parts = strsplit(edgeKeys{e}, '-');
        edges(e,:) = [str2double(parts{1}), str2double(parts{2})];
    end
    
    fprintf('  ✓ Created %d edges\n', size(edges,1));
end


%% ===================================================================
%% CONNECTED COMPONENTS
%% ===================================================================

function [components, sizes] = findConnectedComponents(numNodes, adjList)
    components = zeros(numNodes, 1);
    currentComponent = 0;
    
    for startNode = 1:numNodes
        if components(startNode) == 0
            currentComponent = currentComponent + 1;
            queue = startNode;
            
            while ~isempty(queue)
                node = queue(1);
                queue(1) = [];
                
                if components(node) == 0
                    components(node) = currentComponent;
                    neighbors = adjList{node};
                    queue = [queue, neighbors(components(neighbors)==0)];
                end
            end
        end
    end
    
    sizes = zeros(currentComponent, 1);
    for c = 1:currentComponent
        sizes(c) = sum(components == c);
    end
end


%% ===================================================================
%% VISUALIZATION
%% ===================================================================

function visualizeDense(nodes, edges, airwayInterior, gridInfo)
    sliceZ = round(gridInfo.gridSize(3)/2);
    
    figure('Units', 'inches', 'Position', [1 1 3.5 3.2], 'Color', 'w');
    hold on;
    
    % Background
    slice = airwayInterior(:, :, sliceZ);
    imagesc(gridInfo.x, gridInfo.y, slice');
    colormap(flipud(gray));
    
    % Draw ALL edges (green pathways)
    fprintf('Visualizing %d pathway edges...\n', size(edges,1));
    for e = 1:size(edges,1)
        plot([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
             [nodes(edges(e,1),2), nodes(edges(e,2),2)], ...
             '-', 'LineWidth', 1.2, 'Color', [0.2 0.8 0.2]);
    end
    
    % Draw nodes (smaller for dense sampling)
    scatter(nodes(:,1), nodes(:,2), 25, [1 0 0], 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
    
    xlabel('\textbf{X (mm)}', 'FontSize', 9, 'FontName', 'Times New Roman', ...
           'FontWeight', 'bold', 'Interpreter', 'latex');
    ylabel('\textbf{Y (mm)}', 'FontSize', 9, 'FontName', 'Times New Roman', ...
           'FontWeight', 'bold', 'Interpreter', 'latex');
    title(sprintf('\\textbf{Dense Lung Roadmap: %d Nodes, %d Paths}', ...
          size(nodes,1), size(edges,1)), ...
          'FontSize', 9, 'FontName', 'Times New Roman', ...
          'FontWeight', 'bold', 'Interpreter', 'latex');
    
    axis equal tight; grid off;
    set(gca, 'YDir', 'normal', 'FontSize', 9, 'FontName', 'Times New Roman', ...
             'FontWeight', 'bold', 'LineWidth', 1, 'Box', 'on');
    
    hold off;
    
    print('roadmap_dense', '-dpng', '-r600');
    
    fprintf('✓ Saved roadmap_dense.png\n');
end


%% ===================================================================
%% HELPER FUNCTIONS
%% ===================================================================

function clear = isPathClear(p1, p2, airwayInterior, gridInfo)
    dist = norm(p2-p1);
    numSamples = max(20, ceil(dist/gridInfo.voxelSize));
    
    t = linspace(0,1,numSamples);
    for i = 1:numSamples
        pt = p1 + t(i)*(p2-p1);
        [ii,jj,kk,v] = world2grid(pt, gridInfo);
        if ~v || ~airwayInterior(ii,jj,kk)
            clear = false;
            return;
        end
    end
    clear = true;
end

function [i,j,k,valid] = world2grid(point, gridInfo)
    i = round((point(1)-gridInfo.minBounds(1))/gridInfo.voxelSize)+1;
    j = round((point(2)-gridInfo.minBounds(2))/gridInfo.voxelSize)+1;
    k = round((point(3)-gridInfo.minBounds(3))/gridInfo.voxelSize)+1;
    valid = i>=1 && i<=gridInfo.gridSize(1) && j>=1 && j<=gridInfo.gridSize(2) && k>=1 && k<=gridInfo.gridSize(3);
end