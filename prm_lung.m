%% ===================================================================
%% K-NEAREST NEIGHBOR ROADMAP (GUARANTEES CONNECTIVITY)
%% ===================================================================

clc; clear; close all;

load('bronchial_obstacle_map.mat', 'obstacleMap', 'gridInfo');
airwayInterior = obstacleMap;

%% PARAMETERS
config.numNodes = 500;
config.k = 8;  % Connect each node to 8 nearest neighbors
config.maxConnectionDist = 50;  % Still check max distance

fprintf('====================================\n');
fprintf('K-NN Roadmap (k=%d nearest neighbors)\n', config.k);
fprintf('====================================\n\n');

%% Sample nodes (NO safety margin - just in airway)
fprintf('Sampling %d nodes...\n', config.numNodes);
nodes = sampleSimple(airwayInterior, gridInfo, config.numNodes);

%% Build K-NN graph
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
fprintf('Min degree: %d, Max degree: %d\n', min(degrees), max(degrees));
fprintf('Isolated nodes: %d\n', sum(degrees==0));

%% Find largest connected component
fprintf('\nFinding connected components...\n');
[components, sizes] = findConnectedComponents(size(nodes,1), adjList);
fprintf('Total components: %d\n', length(sizes));
fprintf('Largest component: %d nodes (%.1f%%)\n', max(sizes), 100*max(sizes)/size(nodes,1));

% Keep only largest component
[~, largestIdx] = max(sizes);
mainComponentNodes = find(components == largestIdx);

fprintf('\nUsing largest connected component: %d nodes\n', length(mainComponentNodes));

% Filter nodes and edges
nodesFiltered = nodes(mainComponentNodes, :);
nodeMap = zeros(size(nodes,1), 1);
nodeMap(mainComponentNodes) = 1:length(mainComponentNodes);

edgesFiltered = [];
for e = 1:size(edges,1)
    if nodeMap(edges(e,1)) > 0 && nodeMap(edges(e,2)) > 0
        edgesFiltered = [edgesFiltered; nodeMap(edges(e,1)), nodeMap(edges(e,2))];
    end
end

% Save
save('prm_roadmap.mat', 'nodesFiltered', 'edgesFiltered', 'config', 'gridInfo', 'airwayInterior');

%% Visualize
visualizeFinal(nodesFiltered, edgesFiltered, airwayInterior, gridInfo);


%% ===================================================================
%% SIMPLE NODE SAMPLING (no safety margin)
%% ===================================================================

function nodes = sampleSimple(airwayInterior, gridInfo, numNodes)
    [iAir, jAir, kAir] = ind2sub(size(airwayInterior), find(airwayInterior));
    
    minX = min(gridInfo.x(iAir)); maxX = max(gridInfo.x(iAir));
    minY = min(gridInfo.y(jAir)); maxY = max(gridInfo.y(jAir));
    minZ = min(gridInfo.z(kAir)); maxZ = max(gridInfo.z(kAir));
    
    nodes = zeros(numNodes, 3);
    count = 0;
    
    for attempt = 1:(numNodes * 1000)
        if count >= numNodes, break; end
        
        point = [minX + rand()*(maxX-minX), ...
                 minY + rand()*(maxY-minY), ...
                 minZ + rand()*(maxZ-minZ)];
        
        [i,j,k,v] = world2grid(point, gridInfo);
        if v && airwayInterior(i,j,k)
            count = count + 1;
            nodes(count,:) = point;
            if mod(count,100)==0
                fprintf('  %d nodes\n', count);
            end
        end
    end
    
    if count < numNodes
        warning('Only sampled %d/%d nodes', count, numNodes);
        nodes = nodes(1:count,:);
    end
end


%% ===================================================================
%% K-NEAREST NEIGHBOR EDGE BUILDING
%% ===================================================================

function edges = buildKNN_Edges(nodes, airwayInterior, gridInfo, k, maxDist)
    numNodes = size(nodes,1);
    edgeSet = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    
    fprintf('  Computing k-NN for each node...\n');
    
    for i = 1:numNodes
        % Find k nearest neighbors
        distances = sqrt(sum((nodes - nodes(i,:)).^2, 2));
        [sortedDist, sortedIdx] = sort(distances);
        
        % Connect to k nearest (excluding self)
        for j = 2:min(k+1, length(sortedIdx))
            neighbor = sortedIdx(j);
            dist = sortedDist(j);
            
            if dist > maxDist
                continue;
            end
            
            % Check collision
            if isPathClear(nodes(i,:), nodes(neighbor,:), airwayInterior, gridInfo)
                % Add edge (avoid duplicates)
                key = sprintf('%d-%d', min(i,neighbor), max(i,neighbor));
                edgeSet(key) = true;
            end
        end
        
        if mod(i,50)==0
            fprintf('    %d/%d nodes (%d edges)\n', i, numNodes, edgeSet.Count);
        end
    end
    
    % Convert to array
    edgeKeys = keys(edgeSet);
    edges = zeros(length(edgeKeys), 2);
    for e = 1:length(edgeKeys)
        parts = strsplit(edgeKeys{e}, '-');
        edges(e,:) = [str2double(parts{1}), str2double(parts{2})];
    end
    
    fprintf('  ✓ Built %d edges\n', size(edges,1));
end


%% ===================================================================
%% CONNECTED COMPONENTS ANALYSIS
%% ===================================================================

function [components, sizes] = findConnectedComponents(numNodes, adjList)
    components = zeros(numNodes, 1);
    currentComponent = 0;
    
    for startNode = 1:numNodes
        if components(startNode) == 0
            % New component - BFS
            currentComponent = currentComponent + 1;
            queue = startNode;
            
            while ~isempty(queue)
                node = queue(1);
                queue(1) = [];
                
                if components(node) == 0
                    components(node) = currentComponent;
                    queue = [queue, adjList{node}];
                end
            end
        end
    end
    
    % Count sizes
    sizes = zeros(currentComponent, 1);
    for c = 1:currentComponent
        sizes(c) = sum(components == c);
    end
end


%% ===================================================================
%% VISUALIZATION
%% ===================================================================

function visualizeFinal(nodes, edges, airwayInterior, gridInfo)
    sliceZ = round(gridInfo.gridSize(3)/2);
    
    figure('Units', 'inches', 'Position', [1 1 3.5 3.2], 'Color', 'w');
    hold on;
    
    slice = airwayInterior(:, :, sliceZ);
    imagesc(gridInfo.x, gridInfo.y, slice');
    colormap(flipud(gray));
    
    % Draw all edges
    for e = 1:size(edges,1)
        plot([nodes(edges(e,1),1), nodes(edges(e,2),1)], ...
             [nodes(edges(e,1),2), nodes(edges(e,2),2)], ...
             '-', 'LineWidth', 1.5, 'Color', [0.2 0.8 0.2]);
    end
    
    % Draw nodes
    scatter(nodes(:,1), nodes(:,2), 40, [1 0 0], 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
    
    xlabel('\textbf{X (mm)}', 'FontSize', 9, 'FontName', 'Times New Roman', ...
           'FontWeight', 'bold', 'Interpreter', 'latex');
    ylabel('\textbf{Y (mm)}', 'FontSize', 9, 'FontName', 'Times New Roman', ...
           'FontWeight', 'bold', 'Interpreter', 'latex');
    title(sprintf('\\textbf{Lung Roadmap: %d Nodes, %d Edges (k-NN)}', ...
          size(nodes,1), size(edges,1)), ...
          'FontSize', 9, 'FontName', 'Times New Roman', ...
          'FontWeight', 'bold', 'Interpreter', 'latex');
    
    axis equal tight; grid off;
    set(gca, 'YDir', 'normal', 'FontSize', 9, 'FontName', 'Times New Roman', ...
             'FontWeight', 'bold', 'LineWidth', 1, 'Box', 'on');
    
    hold off;
    
    print('roadmap_knn', '-dpng', '-r600');
    
    fprintf('\n✓ Saved roadmap_knn.png\n');
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