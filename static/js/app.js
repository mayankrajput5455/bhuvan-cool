// -------------------------------------------------------------
// Lucknow UHI Cooler - Frontend Interactive Application Logic
// -------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    // App State
    let activeCity = "lucknow";  // Active city selector parameter
    let cityData = null;         // Original baseline data from backend
    let currentGrid = null;      // Active grid state (can be modified by sim/opt/proj)
    let selectedZone = null;     // Selected zone object
    let currentView = "lst_day";  // Active map layer: lst_day, lst_night, hvi, cluster, ventilation
    let activeInterventions = {}; // Track user simulated changes: {zone_id: {tree_planting: X, ...}}
    let shapChart = null;        // Chart.js instance for SHAP values
    let map = null;              // Leaflet Map instance
    let gridLayer = null;        // L.layerGroup for grid rectangles
    let corridorLayer = null;    // L.layerGroup for wind corridors
    let boundaryLayer = null;    // L.layerGroup for city boundary curves
    let highlightLayer = null;   // L.layerGroup for selected/hovered indicators
    
    // Performance optimization structures
    let cachedGridMinLat = 0;
    let cachedGridMaxLat = 0;
    let cachedGridMinLon = 0;
    let cachedGridMaxLon = 0;
    let cachedGridSize = 120;
    let gridLookupMap = {};      // 2D lookup map: gridLookupMap[y][x]
    let zoneIdMap = {};          // ID lookup map: zoneIdMap[zone_id]
    let heatmapOverlay = null;   // Single Leaflet ImageOverlay for the entire grid
    
    const citySelectEl = document.getElementById("city-select");
    
    // UI Selectors
    const mapEl = document.getElementById("map");
    const viewButtons = document.querySelectorAll(".view-btn");
    const legendEl = document.getElementById("map-legend");
    const toggleRiverCheckbox = document.getElementById("toggle-river");
    const toggleCorridorsCheckbox = document.getElementById("toggle-corridors");
    
    // Zone Details Panel Selectors
    const detailsPlaceholder = document.getElementById("details-placeholder-body");
    const detailsActive = document.getElementById("details-active-body");
    const labelZoneId = document.getElementById("selected-zone-id");
    const labelZoneName = document.getElementById("selected-zone-name");
    const labelZoneDesc = document.getElementById("selected-zone-desc");
    const valLstDay = document.getElementById("val-lst-day");
    const valLstNight = document.getElementById("val-lst-night");
    const valHvi = document.getElementById("val-hvi");
    const valNdvi = document.getElementById("val-ndvi");
    const valAlbedo = document.getElementById("val-albedo");
    const valIsf = document.getElementById("val-isf");
    
    // Cooling Strategies Card Selectors
    const strategiesPlaceholder = document.getElementById("strategies-placeholder-body");
    const strategiesActive = document.getElementById("strategies-active-body");
    const strategyCityTag = document.getElementById("strategy-city-tag");
    const strategyZoneName = document.getElementById("strategy-zone-name");
    const strategyZoneSuitability = document.getElementById("strategy-zone-suitability");
    const strategyList = document.getElementById("strategy-list");
    
    // Climate Projection Selectors
    const climateRadios = document.querySelectorAll("input[name='climate-year']");
    
    // Initialize App Data
    fetchCityData();

    // -------------------------------------------------------------
    // 1. Data Fetching
    // -------------------------------------------------------------
    function updateGridMaps() {
        if (!currentGrid || currentGrid.length === 0) return;
        
        cachedGridSize = cityData && cityData.grid_size ? cityData.grid_size : 120;
        
        const lats = currentGrid.map(z => z.latitude);
        const lons = currentGrid.map(z => z.longitude);
        cachedGridMinLat = Math.min(...lats);
        cachedGridMaxLat = Math.max(...lats);
        cachedGridMinLon = Math.min(...lons);
        cachedGridMaxLon = Math.max(...lons);
        
        gridLookupMap = {};
        zoneIdMap = {};
        
        currentGrid.forEach(zone => {
            if (gridLookupMap[zone.y] === undefined) {
                gridLookupMap[zone.y] = {};
            }
            gridLookupMap[zone.y][zone.x] = zone;
            
            zoneIdMap[zone.zone_id] = zone;
        });
    }

    const cityCenters = {
        lucknow: [26.8467, 80.9462],
        delhi: [28.6139, 77.2090],
        kanpur: [26.4499, 80.3319],
        goa: [15.4909, 73.8278],
        mumbai: [19.0760, 72.8777]
    };

    function fetchCityData() {
        fetch("/api/city-data?city=" + activeCity)
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    cityData = data;
                    currentGrid = JSON.parse(JSON.stringify(data.zones)); // Deep clone
                    
                    // Update UI titles dynamically
                    document.getElementById("header-city-badge").innerText = data.city.toUpperCase();
                    document.getElementById("map-city-name").innerText = data.city;
                    document.getElementById("diagnostics-city-name").innerText = data.city;
                    document.getElementById("projection-city-title").innerText = data.city + " 2050 Projection Engine";
                    
                    updateProjectionDescription("2026");
                    
                    // Reset local overrides
                    activeInterventions = {};
                    currentGrid.forEach(z => {
                        activeInterventions[z.zone_id] = {
                            tree_planting: 0,
                            green_roofs: 0,
                            cool_pavement: 0
                        };
                    });
                    
                    updateGridMaps();
                    heatmapOverlay = null; // Recreate heatmap overlay for the new city
                    
                    // Leaflet map setup
                    if (!map) {
                        map = L.map('map', {
                            zoomControl: true,
                            minZoom: 10,
                            maxZoom: 15,
                            preferCanvas: true
                        });
                        
                        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
                        }).addTo(map);
                        
                        gridLayer = L.layerGroup().addTo(map);
                        corridorLayer = L.layerGroup().addTo(map);
                        boundaryLayer = L.layerGroup().addTo(map);
                        highlightLayer = L.layerGroup().addTo(map);
                    }
                    
                    const center = cityCenters[activeCity] || [26.8467, 80.9462];
                    map.setView(center, activeCity === "mumbai" ? 11 : 12);
                    
                    renderGrid();
                    updateLegend();
                } else {
                    console.error("API error:", data.message);
                }
            })
            .catch(err => {
                console.error("Fetch error:", err);
            });
    }
    
    // Hook up city select change listener
    if (citySelectEl) {
        citySelectEl.addEventListener("change", (e) => {
            activeCity = e.target.value;
            selectedZone = null;
            detailsPlaceholder.classList.remove("hidden");
            detailsActive.classList.add("hidden");
            // Reset strategies panel state
            strategiesPlaceholder.classList.remove("hidden");
            strategiesActive.classList.add("hidden");
            
            // Reset projections to baseline
            document.querySelector("input[name='climate-year'][value='2026']").checked = true;
            fetchCityData();
        });
    }

    // -------------------------------------------------------------
    // 2. Map Rendering (Grid Map)
    // -------------------------------------------------------------
    // Nearest zone Euclidean search helper
    function getNearestZone(lat, lon) {
        if (!currentGrid || !gridLookupMap) return null;
        
        const rowFraction = (lat - cachedGridMinLat) / (cachedGridMaxLat - cachedGridMinLat);
        const colFraction = (lon - cachedGridMinLon) / (cachedGridMaxLon - cachedGridMinLon);
        
        const row = Math.round(rowFraction * (cachedGridSize - 1));
        const col = Math.round(colFraction * (cachedGridSize - 1));
        
        const clampedRow = Math.max(0, Math.min(cachedGridSize - 1, row));
        const clampedCol = Math.max(0, Math.min(cachedGridSize - 1, col));
        
        if (gridLookupMap[clampedRow] && gridLookupMap[clampedRow][clampedCol]) {
            return gridLookupMap[clampedRow][clampedCol];
        }
        
        return null;
    }

    let hoverCircle = null;
    function drawHoverIndicator(zone) {
        if (!map) return;
        if (hoverCircle) {
            highlightLayer.removeLayer(hoverCircle);
        }
        
        const val = getMetricValue(zone, currentView);
        const color = zone.is_water ? "#004b6b" : getColorForValue(val, currentView, zone);
        
        hoverCircle = L.circle([zone.latitude, zone.longitude], {
            radius: 250, // 250m radius
            color: color,
            weight: 1.5,
            fillColor: color,
            fillOpacity: 0.3,
            dashArray: "4, 4"
        }).addTo(highlightLayer);
    }
    
    function clearHoverIndicator() {
        if (hoverCircle && map) {
            highlightLayer.removeLayer(hoverCircle);
            hoverCircle = null;
        }
    }

    let selectedCircle = null;
    let selectedOuterRing = null;
    function drawSelectedHighlight(zone) {
        if (!map) return;
        
        if (selectedCircle) highlightLayer.removeLayer(selectedCircle);
        if (selectedOuterRing) highlightLayer.removeLayer(selectedOuterRing);
        
        const val = getMetricValue(zone, currentView);
        const color = zone.is_water ? "#00d2ff" : getColorForValue(val, currentView, zone);
        
        // Solid center circle
        selectedCircle = L.circle([zone.latitude, zone.longitude], {
            radius: 350,
            color: "var(--accent-teal)",
            weight: 2,
            fillColor: color,
            fillOpacity: 0.45
        }).addTo(highlightLayer);
        
        // Pulsing/dashed outer ring
        selectedOuterRing = L.circle([zone.latitude, zone.longitude], {
            radius: 500,
            color: "var(--accent-teal)",
            weight: 1,
            fill: false,
            dashArray: "6, 6"
        }).addTo(highlightLayer);
    }

    function renderGrid() {
        if (!map || !currentGrid) return;
        
        boundaryLayer.clearLayers();
        highlightLayer.clearLayers();
        
        const gridSize = cityData && cityData.grid_size ? cityData.grid_size : 120;
        
        // 1. Re-draw the boundary polygon
        const dxs = currentGrid.map(z => z.latitude);
        const dys = currentGrid.map(z => z.longitude);
        const minLat = Math.min(...dxs);
        const maxLat = Math.max(...dxs);
        const minLon = Math.min(...dys);
        const maxLon = Math.max(...dys);
        
        const centerLat = (minLat + maxLat) / 2;
        const centerLon = (minLon + maxLon) / 2;
        const halfLat = (maxLat - minLat) / 2;
        const halfLon = (maxLon - minLon) / 2;
        
        const padding = 1.08;
        const a = halfLon * padding;
        const b = halfLat * padding;
        
        const boundaryPoints = [];
        const steps = 72;
        const n = 3.5;
        
        for (let i = 0; i < steps; i++) {
            const t = (i / steps) * 2 * Math.PI;
            const cosT = Math.cos(t);
            const sinT = Math.sin(t);
            
            const x = Math.sign(cosT) * Math.pow(Math.abs(cosT), 2 / n) * a;
            const y = Math.sign(sinT) * Math.pow(Math.abs(sinT), 2 / n) * b;
            
            boundaryPoints.push([centerLat + y, centerLon + x]);
        }
        
        const boundaryPoly = L.polygon(boundaryPoints, {
            color: "var(--accent-teal)",
            weight: 2,
            opacity: 0.85,
            fillColor: "rgba(0, 242, 254, 0.01)",
            fillOpacity: 0.1,
            fill: true,
            lineJoin: "round",
            className: "city-boundary-poly"
        }).addTo(boundaryLayer);
        
        boundaryPoly.bindTooltip("", { sticky: true, className: "leaflet-tooltip-custom" });
        
        boundaryPoly.on("mousemove", (e) => {
            const nearest = getNearestZone(e.latlng.lat, e.latlng.lng);
            if (nearest) {
                const val = getMetricValue(nearest, currentView);
                let metricName = "";
                let metricValStr = "";
                if (currentView === "lst_day") {
                    metricName = "LST (Day)";
                    metricValStr = `${val.toFixed(1)}°C`;
                } else if (currentView === "lst_night") {
                    metricName = "LST (Night)";
                    metricValStr = `${val.toFixed(1)}°C`;
                } else if (currentView === "hvi") {
                    metricName = "HVI";
                    metricValStr = nearest.is_water ? "N/A" : `${val.toFixed(2)} (${nearest.hvi_class})`;
                } else if (currentView === "cluster") {
                    metricName = "Priority";
                    metricValStr = nearest.cluster_label;
                } else if (currentView === "ventilation") {
                    metricName = "Ventilation Suitability";
                    metricValStr = val.toFixed(2);
                }
                
                let tooltipText = `<strong>${nearest.name}</strong><br>${metricName}: ${metricValStr}`;
                if (nearest.is_water) {
                    tooltipText = `<strong>${nearest.name}</strong><br>Water Zone<br>LST: ${getLSTValue(nearest).toFixed(1)}°C`;
                }
                boundaryPoly.setTooltipContent(tooltipText);
                drawHoverIndicator(nearest);
            }
        });
        
        boundaryPoly.on("mouseout", () => {
            clearHoverIndicator();
        });
        
        boundaryPoly.on("click", (e) => {
            const nearest = getNearestZone(e.latlng.lat, e.latlng.lng);
            if (nearest) {
                selectZone(nearest.zone_id);
            }
        });
        
        // 2. Render Heatmap Image Overlay
        const latStep = (maxLat - minLat) / (gridSize - 1);
        const lonStep = (maxLon - minLon) / (gridSize - 1);
        const bounds = [
            [minLat - latStep / 2, minLon - lonStep / 2],
            [maxLat + latStep / 2, maxLon + lonStep / 2]
        ];
        
        const canvas = document.createElement("canvas");
        canvas.width = gridSize;
        canvas.height = gridSize;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, gridSize, gridSize);
        
        const showWaterGeometry = toggleRiverCheckbox.checked;
        
        currentGrid.forEach(zone => {
            const x = zone.x;
            const y = (gridSize - 1) - zone.y; // Invert y-axis for canvas
            
            if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) return;
            
            const dx = (zone.longitude - centerLon) / a;
            const dy = (zone.latitude - centerLat) / b;
            const dist = Math.pow(Math.abs(dx), n) + Math.pow(Math.abs(dy), n);
            
            if (dist <= 1.05) {
                if (zone.is_water) {
                    if (showWaterGeometry) {
                        ctx.fillStyle = "#004b6b";
                        ctx.globalAlpha = 0.35;
                        ctx.fillRect(x, y, 1, 1);
                    }
                } else {
                    const val = getMetricValue(zone, currentView);
                    ctx.fillStyle = getColorForValue(val, currentView, zone);
                    ctx.globalAlpha = 0.65;
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        });
        
        const dataUrl = canvas.toDataURL();
        
        if (heatmapOverlay) {
            heatmapOverlay.setUrl(dataUrl);
            heatmapOverlay.setBounds(bounds);
        } else {
            gridLayer.clearLayers(); // Clear old city overlays
            heatmapOverlay = L.imageOverlay(dataUrl, bounds, {
                opacity: 1.0,
                interactive: false,
                className: "heatmap-image-overlay"
            });
            gridLayer.addLayer(heatmapOverlay);
        }
        
        // 3. Highlight selected zone
        if (selectedZone) {
            drawSelectedHighlight(selectedZone);
        }
        
        // 4. Update wind corridors
        drawWindCorridors();
    }

    // Extract value for active view
    function getMetricValue(zone, view) {
        if (view === "lst_day") {
            return getLSTValue(zone);
        } else if (view === "lst_night") {
            // If nighttime pred exists, use it, else baseline
            return zone.lst_night_pred !== undefined ? zone.lst_night_pred : zone.lst_night_actual;
        } else if (view === "hvi") {
            return zone.hvi;
        } else if (view === "cluster") {
            return zone.cluster;
        } else if (view === "ventilation") {
            return zone.ventilation_suitability;
        }
        return 0;
    }

    // Helper to get predicted day temperature or actual baseline
    function getLSTValue(zone) {
        return zone.lst_day_pred !== undefined ? zone.lst_day_pred : zone.lst_day_actual;
    }

    // Wind corridor list helper
    function isWindCorridorZone(zoneId) {
        if (!cityData || !cityData.corridors) return false;
        // Check if zone is in any of the corridor lists
        return cityData.corridors.some(c => c.zones.includes(zoneId));
    }

    // Helper to calculate cell color
    function getColorForValue(val, view, zone) {
        if (view === "lst_day") {
            // Scale between 30°C (Vibrant Blue/Cool) and 70°C (Extreme Heat/Red)
            const minT = 30.0;
            const maxT = 70.0;
            const norm = Math.max(0, Math.min(1, (val - minT) / (maxT - minT)));
            // Spectral scale: 240 (Blue) -> 120 (Green) -> 60 (Yellow) -> 0 (Red)
            const hue = (1.0 - norm) * 240;
            return `hsl(${hue}, 95%, 48%)`;
        } 
        else if (view === "lst_night") {
            // Scale between 20°C and 36°C
            const minT = 20.0;
            const maxT = 36.0;
            const norm = Math.max(0, Math.min(1, (val - minT) / (maxT - minT)));
            const hue = (1.0 - norm) * 240;
            return `hsl(${hue}, 95%, 48%)`;
        } 
        else if (view === "hvi") {
            // Scale between 0.15 (Green/Safe) and 0.85 (Extreme/Red)
            const minH = 0.20;
            const maxH = 0.80;
            const norm = Math.max(0, Math.min(1, (val - minH) / (maxH - minH)));
            const hue = (1.0 - norm) * 120;
            return `hsl(${hue}, 75%, 45%)`;
        } 
        else if (view === "cluster") {
            // Discrete colors for K-Means clusters (Priority classes)
            const clusterColors = {
                0: "#2ed573", // Green (Cool & Resilient)
                1: "#1e90ff", // Blue (Moderate Risk)
                2: "#ffa502", // Orange (Heat Exposed)
                3: "#ff4757"  // Red (Critical Action)
            };
            return clusterColors[val] || "#1e293b";
        } 
        else if (view === "ventilation") {
            // Color map for wind corridors: dark navy (0) to light cyan/teal (1)
            const norm = Math.max(0, Math.min(1, val));
            return `rgba(0, 245, 212, ${norm * 0.9 + 0.1})`;
        }
        return "#1e293b";
    }

    // -------------------------------------------------------------
    // 3. Setup SVG Overlay Layers (Gomti River & Wind Pathways)
    // -------------------------------------------------------------
    function drawWindCorridors() {
        if (!map || !cityData) return;
        corridorLayer.clearLayers();
        
        const isCorridorActive = toggleCorridorsCheckbox.checked;
        if (!isCorridorActive || !cityData.corridors) return;
        
        cityData.corridors.forEach(corridor => {
            const latlngs = corridor.zones
                .map(zoneId => zoneIdMap[zoneId])
                .filter(z => z !== undefined)
                .map(z => [z.latitude, z.longitude]);
            
            if (latlngs.length < 2) return;
            
            L.polyline(latlngs, {
                className: "corridor-line",
                opacity: 0.8
            }).addTo(corridorLayer);
        });
    }

    toggleRiverCheckbox.addEventListener("change", () => {
        renderGrid();
    });
    toggleCorridorsCheckbox.addEventListener("change", () => {
        renderGrid();
    });

    // -------------------------------------------------------------
    // 4. View Switcher handling
    // -------------------------------------------------------------
    viewButtons.forEach(btn => {
        btn.addEventListener("click", (e) => {
            viewButtons.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            
            currentView = e.target.getAttribute("data-view");
            renderGrid();
            updateLegend();
        });
    });

    function updateLegend() {
        legendEl.innerHTML = "";
        
        if (currentView === "lst_day" || currentView === "lst_night") {
            const minT = currentView === "lst_day" ? "30°C" : "20°C";
            const maxT = currentView === "lst_day" ? "70°C" : "36°C";
            legendEl.innerHTML = `
                <div class="legend-item"><span class="legend-color" style="background: hsl(240, 95%, 48%)"></span> <span>Cool (${minT})</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(180, 95%, 48%)"></span> <span>Mild</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(120, 95%, 48%)"></span> <span>Moderate</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(60, 95%, 48%)"></span> <span>Warm</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(0, 95%, 48%)"></span> <span>Extreme Heat (${maxT})</span></div>
            `;
        } 
        else if (currentView === "hvi") {
            legendEl.innerHTML = `
                <div class="legend-item"><span class="legend-color" style="background: hsl(120, 75%, 45%)"></span> <span>Low Vulnerability</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(75, 75%, 45%)"></span> <span>Moderate</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(35, 75%, 45%)"></span> <span>High</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(0, 75%, 45%)"></span> <span>Extreme Risk</span></div>
            `;
        } 
        else if (currentView === "cluster") {
            legendEl.innerHTML = `
                <div class="legend-item"><span class="legend-color" style="background: #2ed573"></span> <span>Cool & Resilient</span></div>
                <div class="legend-item"><span class="legend-color" style="background: #1e90ff"></span> <span>Moderate Risk</span></div>
                <div class="legend-item"><span class="legend-color" style="background: #ffa502"></span> <span>Heat Exposed</span></div>
                <div class="legend-item"><span class="legend-color" style="background: #ff4757"></span> <span>Critical Action Zone</span></div>
            `;
        } 
        else if (currentView === "ventilation") {
            legendEl.innerHTML = `
                <div class="legend-item"><span class="legend-color" style="background: rgba(0, 245, 212, 0.1)"></span> <span>Low Suitability (High building roughness)</span></div>
                <div class="legend-item"><span class="legend-color" style="background: rgba(0, 245, 212, 0.5)"></span> <span>Medium Flow</span></div>
                <div class="legend-item"><span class="legend-color" style="background: rgba(0, 245, 212, 0.9)"></span> <span>Active Cool Wind Corridor</span></div>
            `;
        }
    }

    // -------------------------------------------------------------
    // 5. Zone Selection & SHAP Charts
    // -------------------------------------------------------------
    function selectZone(zoneId) {
        const found = zoneIdMap[zoneId];
        if (!found) return;
        selectedZone = found;
        
        // Highlight active cell on map via pulsing circles
        drawSelectedHighlight(selectedZone);
        
        // Render details panel
        detailsPlaceholder.classList.add("hidden");
        detailsActive.classList.remove("hidden");
        
        labelZoneId.innerText = selectedZone.zone_id.toUpperCase();
        labelZoneName.innerText = selectedZone.name;
        
        // Description helper
        const landmarkName = Object.keys(mp_landmark_descs).find(k => selectedZone.name.includes(k));
        if (selectedZone.is_water) {
            labelZoneDesc.innerText = "Open water body. Directly contributes to sea-breeze microclimate cooling. Human intervention and urban landscaping cannot be executed on water zones.";
        } else {
            labelZoneDesc.innerText = landmarkName ? mp_landmark_descs[landmarkName] : "Urban residential zone.";
        }
        
        const tempVal = getLSTValue(selectedZone);
        valLstDay.innerText = `${tempVal.toFixed(1)}°C`;
        valLstNight.innerText = `${(selectedZone.lst_night_pred !== undefined ? selectedZone.lst_night_pred : selectedZone.lst_night_actual).toFixed(1)}°C`;
        
        // Set HVI badge and class coloring
        if (selectedZone.is_water) {
            valHvi.innerText = "N/A (Water Cell)";
            valHvi.style.backgroundColor = "rgba(0, 210, 255, 0.15)";
            valHvi.style.color = "var(--accent-blue)";
        } else {
            valHvi.innerText = `${selectedZone.hvi.toFixed(2)} (${selectedZone.hvi_class})`;
            valHvi.className = "metric-value hvi-badge"; 
            if (selectedZone.hvi_class === "Low Risk") valHvi.style.backgroundColor = "rgba(46, 213, 115, 0.2)", valHvi.style.color = "#2ed573";
            else if (selectedZone.hvi_class === "Medium Risk") valHvi.style.backgroundColor = "rgba(30, 144, 255, 0.2)", valHvi.style.color = "#1e90ff";
            else if (selectedZone.hvi_class === "High Risk") valHvi.style.backgroundColor = "rgba(255, 165, 2, 0.2)", valHvi.style.color = "#ffa502";
            else valHvi.style.backgroundColor = "rgba(255, 71, 87, 0.2)", valHvi.style.color = "#ff4757";
        }
        
        valNdvi.innerText = selectedZone.ndvi.toFixed(2);
        valAlbedo.innerText = selectedZone.albedo.toFixed(2);
        valIsf.innerText = selectedZone.is_water ? "0% (Water)" : `${Math.round(selectedZone.isf * 100)}%`;
        
        // Update cooling strategies
        updateCoolingStrategies(selectedZone);
        
        // Draw SHAP explanation
        fetchAndDrawShap(selectedZone.zone_id);
    }

    // Heuristic Landmark Description mapping for details panel
    const mp_landmark_descs = {
        // Lucknow
        "Chowk": "Extremely dense old city market. High building fraction, brick/masonry structures, narrow alleys, high thermal mass, and almost zero vegetative cover.",
        "Hazratganj": "Central commercial business district of Lucknow. High paving fraction (asphalt roads, plazas), concrete facades, intense daytime traffic and AC waste heat loading.",
        "Gomti Nagar": "Modern planned residential community with extensive park reserves, grass verges, and roadside tree canopy corridors. Benefits from nearby Gomti River winds.",
        "Kukrail Forest": "State reserve forest area on Lucknow's northeastern fringe. Extensive dense multi-layer broadleaf tree cover providing excellent evapotranspirative cooling.",
        "Aminabad": "One of India's densest historic retail centers. Highly compact building layout, massive human heat load, paved roads, and negligible cooling corridors.",
        "Charbagh": "Major railway junction with heavy paved asphalt grids, minimal cover, high metal roof thermal absorption, and intense heavy vehicular emissions.",
        "Alambagh": "Industrial and commercial sector flanking highway corridors. High road density, metal roofing elements, and limited localized green spaces.",
        "Indiranagar": "Established mid-density residential area with moderate park blocks and mature roadside tree canopy alignments.",
        "Jankipuram": "Suburban block in North Lucknow undergoing development. Medium building density, open soils, and developing layout.",
        "Vrindavan Yojana": "Southern expansion zone with bare soil clearing, new concrete apartments, and high solar exposure.",
        
        // Delhi
        "Chandni Chowk": "Historic ultra-dense bazaar area. Heavy masonry structures, narrow streets creating heat trapping, massive pedestrian traffic, and zero vegetation.",
        "Connaught Place": "Colonial-style concentric commercial hub. Vast paved concrete, asphalt avenues, high vehicle density, and intense heat dissipation from air conditioning systems.",
        "Delhi Ridge": "Protected scrub forest area on rocky terrain. Provides excellent evaporative cooling buffers for surrounding neighborhoods.",
        "Okhla": "Industrial manufacturing sector. Dark asphalt roads, massive sheet-metal industrial roofs, heavy machinery emissions, and very high daytime temperatures.",
        "Dwarka": "Planned high-density residential township. Large block buildings, asphalt parking lots, with pockets of community gardens.",
        
        // Kanpur
        "Jajmau": "Industrial tanneries block on the Ganges. Dense factory spaces, metal rooftops, dust emissions, and high local heat output.",
        "Civil Lines": "Historic downtown mixed residential sector. Highly compact commercial spaces and paved roads.",
        "Kalyanpur": "Suburban residential development near the outskirts, displaying open sand/soil exposure and moderate green patches.",
        "IIT Kanpur": "Densely forested institutional campus. Wide tree canopies, open lawns, and low paved fraction creating a cool forest microclimate.",
        
        // Goa
        "Panaji": "Coastal urban center. Historic Portuguese buildings with red terracotta tile roofs, narrow streets, and direct breeze influence.",
        "Miramar": "Sandy beachfront displaying very high albedo. High solar reflectivity and direct cooling sea breeze blocks.",
        "Saligao": "Tropical broadleaf forest canopy, dense greenery, creating strong local transpiration buffers.",
        "Vasco": "Industrial port area. Large metal warehouses, shipping container stacks, asphalt lots, and heavy loading machinery heat.",
        
        // Mumbai
        "Dharavi": "Informal residential block of extreme density. Corrugated iron rooftops, compact pathways, zero vegetation, and high heat capacity.",
        "Nariman": "High-rise business district at the southern tip. Dense skyscraper block layout surrounded by sea breezes on three sides.",
        "Sanjay Gandhi": "Vast national forest reserve in northern Mumbai. Dense tropical vegetation creating a major regional cool island.",
        "Bandra": "Modern business and residential district (BKC). High concrete coverage, glass buildings, and intense microclimate trapping.",
        "Alambagh": "Mixed commercial and residential sector flanking heavy highway corridors. Medium density with industrial warehouse roofs creating localized heat pockets.",
        "Indiranagar": "Established mid-density residential area with moderate residential garden patches, mature trees, and concrete building blocks.",
        "Jankipuram": "Suburban township in northern Lucknow undergoing expansion. Moderate vegetation density with new paved residential zones.",
        "Vrindavan Yojana": "Southern suburban development area with massive ongoing construction activity, bare soils, and high concrete exposure."
    };

    // -------------------------------------------------------------
    // 6. SHAP Explainer Chart Rendering
    // -------------------------------------------------------------
    function fetchAndDrawShap(zoneId) {
        fetch(`/api/shap/${zoneId}?city=${activeCity}`)
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    renderShapChart(data.explanation);
                }
            })
            .catch(err => console.error("Error loading SHAP values:", err));
    }

    function renderShapChart(explanation) {
        const labels = explanation.map(item => item.label);
        const dataValues = explanation.map(item => item.shap_contribution);
        
        // Color mapping: red/warm color for positive impact (heating driver), green/cool color for negative (cooling driver)
        const backgroundColors = dataValues.map(val => val >= 0 ? "rgba(255, 71, 87, 0.85)" : "rgba(46, 213, 115, 0.85)");
        const borderColors = dataValues.map(val => val >= 0 ? "#ff4757" : "#2ed573");
        
        if (shapChart) {
            shapChart.destroy();
        }
        
        const ctx = document.getElementById("shapChart").getContext("2d");
        shapChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Temperature Impact (°C)",
                    data: dataValues,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.raw;
                                return `${val >= 0 ? '+' : ''}${val.toFixed(2)} °C LST driver`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255,255,255,0.06)" },
                        ticks: { color: "#94a3b8", font: { size: 9 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#f1f5f9", font: { size: 10 } }
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------
    // 7. Targeted Cooling Strategies Generator
    // -------------------------------------------------------------
    function updateCoolingStrategies(zone) {
        if (!zone) return;
        
        strategiesPlaceholder.classList.add("hidden");
        strategiesActive.classList.remove("hidden");
        
        strategyCityTag.innerText = activeCity.toUpperCase() + " STRATEGY";
        strategyZoneName.innerText = zone.name;
        
        if (zone.is_water) {
            strategyZoneSuitability.innerText = "Natural Water Body Zone - Active Heat Sink";
            strategyList.innerHTML = `
                <div class="strategy-item priority-low">
                    <span class="strategy-icon">🌊</span>
                    <div class="strategy-content">
                        <span class="strategy-title">Protect Natural Blue Infrastructure</span>
                        <span class="strategy-desc">This is an open water grid zone. It plays a critical role in cooling via evaporation and creating localized sea/river breezes. Human interventions or urban development are blocked to safeguard the water microclimate.</span>
                    </div>
                </div>
                <div class="strategy-item priority-medium">
                    <span class="strategy-icon">💨</span>
                    <div class="strategy-content">
                        <span class="strategy-title">Maintain Ventilation Lanes</span>
                        <span class="strategy-desc">Ensure that surrounding land zones maintain open wind corridors. Restrict high-rise building density directly adjacent to the water front to prevent blocking cool air transport inland.</span>
                    </div>
                </div>
            `;
            return;
        }
        
        // Land zone suitability score display
        const suitabilityScore = ((1.0 - zone.isf) * 50 + (zone.wind_speed / 6.0) * 30 + (zone.ndvi) * 20).toFixed(0);
        strategyZoneSuitability.innerText = `Cooling Suitability Score: ${suitabilityScore}/100`;
        
        let strategies = [];
        
        // 1. NDVI Strategy
        if (zone.ndvi < 0.20) {
            strategies.push({
                priority: "high",
                icon: "🌳",
                title: "Aggressive Urban Tree Planting",
                desc: `Vegetation cover is critically low (NDVI: ${zone.ndvi.toFixed(2)}). Implement street tree lines, pocket forests, and public parks to increase evapotranspiration. Target NDVI increase to 0.35+.`
            });
        } else if (zone.ndvi < 0.40) {
            strategies.push({
                priority: "medium",
                icon: "🌿",
                title: "Vegetative Buffer Extensions",
                desc: `Moderate vegetation cover (NDVI: ${zone.ndvi.toFixed(2)}). Extend tree rows, introduce vertical greening on building facades, and establish community gardens.`
            });
        } else {
            strategies.push({
                priority: "low",
                icon: "🌲",
                title: "Preserve Existing Green Cover",
                desc: `Excellent vegetation density (NDVI: ${zone.ndvi.toFixed(2)}). Protect the mature tree canopy from real-estate clearing and maintain local biodiversity.`
            });
        }
        
        // 2. Albedo Strategy
        if (zone.albedo < 0.15) {
            strategies.push({
                priority: "high",
                icon: "🎨",
                title: "Cool Roofs and High-Albedo Coatings",
                desc: `Surface albedo is low (Reflectance: ${zone.albedo.toFixed(2)}), leading to high solar heat absorption. Apply white elastomeric or reflective roof coatings on flat rooftops.`
            });
        } else {
            strategies.push({
                priority: "low",
                icon: "🏠",
                title: "Cool Surface Maintenance",
                desc: `Good solar reflectivity (Albedo: ${zone.albedo.toFixed(2)}). Ensure periodic cleaning of high-albedo roofs to maintain optimal reflectance and minimize solar heating.`
            });
        }
        
        // 3. ISF Strategy
        if (zone.isf > 0.70) {
            strategies.push({
                priority: "high",
                icon: "🧱",
                title: "De-Paving & Permeable Surfaces",
                desc: `High impervious fraction (ISF: ${Math.round(zone.isf * 100)}%). Replace asphalt parking lots and pedestrian pathways with permeable grass-pavements or light-colored gravel to reduce heat retention.`
            });
        } else if (zone.isf > 0.40) {
            strategies.push({
                priority: "medium",
                icon: "🏡",
                title: "Green Roof Installation",
                desc: `Moderate impervious surface (ISF: ${Math.round(zone.isf * 100)}%). Install intensive or extensive green roofs on commercial and municipal buildings to store stormwater and cool the air.`
            });
        }
        
        // 4. HVI Strategy
        if (zone.hvi >= 0.68) {
            strategies.push({
                priority: "high",
                icon: "🌡️",
                title: "Extreme Social Heat Support",
                desc: `This locality is an extreme heat-vulnerability risk zone (HVI Score: ${zone.hvi.toFixed(2)}). Establish designated public cooling centers, deploy misting fans in public markets, and run regular heat safety drills.`
            });
        } else if (zone.hvi >= 0.52) {
            strategies.push({
                priority: "medium",
                icon: "💧",
                title: "Community Hydration Stations",
                desc: `High vulnerability area (HVI Score: ${zone.hvi.toFixed(2)}). Provide free drinking water kiosks, public shade structures, and coordinate cooling support networks for the elderly.`
            });
        }
        
        // 5. Proximity to River/Ventilation
        if (zone.distance_to_river < 2.0) {
            strategies.push({
                priority: "medium",
                icon: "💨",
                title: "Ventilation Path Protection",
                desc: `Zone is close to a major wind ventilation river/sea corridor. Enforce building height limits and open setbacks perpendicular to the water front to allow cool breeze penetration.`
            });
        }
        
        // 6. City-Specific Regional Strategies
        let regionalTitle = "";
        let regionalDesc = "";
        let regionalIcon = "📍";
        
        if (activeCity === "lucknow") {
            regionalTitle = "Lucknow Green-Blue Integration";
            regionalDesc = "Align green corridors connecting local parks with the Gomti River basin. Promote traditional high-ventilation courtyard layouts in dense retail markets like Aminabad.";
            regionalIcon = "🕌";
        } else if (activeCity === "delhi") {
            regionalTitle = "Delhi Ridge Buffer Enforcement";
            regionalDesc = "Enforce dense scrub-forest planting around the Delhi Ridge boundary. Combine heat mitigation with dust-arresting vegetative barriers along industrial perimeters in Okhla.";
            regionalIcon = "🏛️";
        } else if (activeCity === "kanpur") {
            regionalTitle = "Kanpur Industrial Heat Barriers";
            regionalDesc = "Plant thick multi-layered tree shelterbelts around industrial tanneries in Jajmau. Coat heavy sheet-metal factory roofs with high-reflectivity solar paints.";
            regionalIcon = "🏭";
        } else if (activeCity === "goa") {
            regionalTitle = "Goa Maritime Wind Setbacks";
            regionalDesc = "Protect shoreline beach setbacks and conserve coastal sand dune vegetation to ensure unobstructed sea-breeze microclimate cooling in beach localities.";
            regionalIcon = "🏖️";
        } else if (activeCity === "mumbai") {
            regionalTitle = "Mumbai Informal Settlement Mitigation";
            regionalDesc = "Scale community-led cool roof painting campaigns in high-density informal settlements (like Dharavi) to alleviate severe indoor heat trapping under metal sheets.";
            regionalIcon = "🏢";
        }
        
        strategies.push({
            priority: "medium",
            icon: regionalIcon,
            title: regionalTitle,
            desc: regionalDesc
        });
        
        // Render strategies
        strategyList.innerHTML = "";
        strategies.forEach(item => {
            const el = document.createElement("div");
            el.className = `strategy-item priority-${item.priority}`;
            el.innerHTML = `
                <span class="strategy-icon">${item.icon}</span>
                <div class="strategy-content">
                    <span class="strategy-title">${item.title}</span>
                    <span class="strategy-desc">${item.desc}</span>
                </div>
            `;
            strategyList.appendChild(el);
        });
    }

    // 8. Grid State Utilities
    // -------------------------------------------------------------
    function resetGridState() {
        if (cityData) {
            currentGrid = JSON.parse(JSON.stringify(cityData.zones));
            activeInterventions = {};
            currentGrid.forEach(z => {
                activeInterventions[z.zone_id] = { tree_planting: 0, green_roofs: 0, cool_pavement: 0 };
            });
            
            updateGridMaps();
            renderGrid();
            
            // Reset strategies panel
            strategiesPlaceholder.classList.remove("hidden");
            strategiesActive.classList.add("hidden");
            
            if (selectedZone) {
                selectZone(selectedZone.zone_id);
            }
        }
    }

    // -------------------------------------------------------------
    // 9. Climate Projections 2050
    // -------------------------------------------------------------
    function updateProjectionDescription(scenario) {
        const descriptions = {
            lucknow: {
                "2026": `<strong>Lucknow today (2026):</strong> Dense urban core with avg LST 38–43°C. Green cover concentrated in Gomti riverbanks.<br><br><strong>Budget:</strong><br>• Green cover maint: ₹18 Cr/yr<br>• Cool pavement trial: ₹6 Cr/yr<br>• No major mitigation active`,
                "2050-rcp45": `<strong>Lucknow 2050 (Moderate):</strong> Avg LST rises ~2°C. Gomti riverfront developed as a cooling corridor.<br><br><strong>Budget:</strong><br>• 15% green cover addition: ₹210 Cr<br>• Cool pavements (albedo ≥0.4): ₹95 Cr<br>• Riverfront restoration: ₹140 Cr<br>• <em>Total: ~₹445 Cr</em>`,
                "2050-rcp85": `<strong>Lucknow 2050 (Extreme):</strong> Avg LST surges ~4.5°C. Vegetation drops below 8%. Gomti cooling effect nearly lost.<br><br><strong>Budget:</strong><br>• Emergency green cover: ₹380 Cr<br>• Cool pavement retrofitting: ₹220 Cr<br>• Heat emergency infra: ₹175 Cr<br>• <em>Total: ~₹775 Cr</em>`
            },
            delhi: {
                "2026": `<strong>Delhi today (2026):</strong> Average LST 40–46°C. Yamuna floodplain provides limited cooling.<br><br><strong>Budget:</strong><br>• Current green dev: ₹45 Cr/yr<br>• Yamuna restoration: ₹28 Cr/yr<br>• Heat action plan: ₹12 Cr/yr`,
                "2050-rcp45": `<strong>Delhi 2050 (Moderate):</strong> LST rises ~2.5°C. 20% tree cover mandate. Yamuna corridor partially revived.<br><br><strong>Budget:</strong><br>• 20% tree cover: ₹520 Cr<br>• Cool roof mandate: ₹180 Cr<br>• Yamuna restoration: ₹310 Cr<br>• <em>Total: ~₹1,010 Cr</em>`,
                "2050-rcp85": `<strong>Delhi 2050 (Extreme):</strong> LST climbs ~5°C. Heatwave frequency triples. Yamuna dries seasonally.<br><br><strong>Budget:</strong><br>• Emergency cooling shelters: ₹290 Cr<br>• Large-scale greening: ₹680 Cr<br>• River revival: ₹450 Cr<br>• <em>Total: ~₹1,420 Cr</em>`
            },
            kanpur: {
                "2026": `<strong>Kanpur today (2026):</strong> Industrial heat + slum clusters push LST to 39–44°C. Ganga riverfront offers narrow cooling.<br><br><strong>Budget:</strong><br>• Industrial compliance: ₹22 Cr/yr<br>• Slum re-roofing pilot: ₹8 Cr/yr`,
                "2050-rcp45": `<strong>Kanpur 2050 (Moderate):</strong> LST rises ~2°C. Industrial norms tightened. Ganga ghat green buffers added.<br><br><strong>Budget:</strong><br>• Cool roofs (resettlement): ₹95 Cr<br>• Ganga ghat greening: ₹160 Cr<br>• Industrial emission control: ₹130 Cr<br>• <em>Total: ~₹385 Cr</em>`,
                "2050-rcp85": `<strong>Kanpur 2050 (Extreme):</strong> LST rises ~4.5°C. Industrial expansion unchecked. Slum heat-exposure critical.<br><br><strong>Budget:</strong><br>• Slum heat-proofing: ₹250 Cr<br>• Forced green buffers: ₹310 Cr<br>• Health infra scaling: ₹180 Cr<br>• <em>Total: ~₹740 Cr</em>`
            },
            goa: {
                "2026": `<strong>Goa today (2026):</strong> Coastal breeze moderates LST to 32–36°C. Mangroves and wooded hills provide natural cooling.<br><br><strong>Budget:</strong><br>• Mangrove conservation: ₹10 Cr/yr<br>• Tourism green compliance: ₹6 Cr/yr`,
                "2050-rcp45": `<strong>Goa 2050 (Moderate):</strong> LST rises ~1.5°C. CRZ enforced. Mangrove restoration along Mandovi & Zuari.<br><br><strong>Budget:</strong><br>• Mangrove restoration: ₹95 Cr<br>• Coastal green buffer: ₹75 Cr<br>• Tourism green mandate: ₹60 Cr<br>• <em>Total: ~₹230 Cr</em>`,
                "2050-rcp85": `<strong>Goa 2050 (Extreme):</strong> LST rises ~3°C. Mangroves degraded by saltwater intrusion. Tourism infrastructure expands inland.<br><br><strong>Budget:</strong><br>• Mangrove revival: ₹180 Cr<br>• Inland green corridors: ₹140 Cr<br>• Coastal protection: ₹210 Cr<br>• <em>Total: ~₹530 Cr</em>`
            },
            mumbai: {
                "2026": `<strong>Mumbai today (2026):</strong> Coastal LST 33–38°C with strong sea-breeze modulation. Eastern suburbs see rapid vertical growth.<br><br><strong>Budget:</strong><br>• Coastal promenade maint: ₹15 Cr/yr<br>• Mangrove patrol: ₹9 Cr/yr`,
                "2050-rcp45": `<strong>Mumbai 2050 (Moderate):</strong> LST rises ~1.5°C. Coastal green promenades expanded. Mangrove cover stabilized.<br><br><strong>Budget:</strong><br>• Promenade expansion: ₹120 Cr<br>• Thane Creek mangrove: ₹90 Cr<br>• Cool pavement coastal road: ₹85 Cr<br>• <em>Total: ~₹295 Cr</em>`,
                "2050-rcp85": `<strong>Mumbai 2050 (Extreme):</strong> LST rises ~3.5°C. Marine heatwaves suppress sea-breeze. Ventilation corridors blocked.<br><br><strong>Budget:</strong><br>• Ventilation corridor clearing: ₹200 Cr<br>• Vertical green retrofit: ₹310 Cr<br>• Coastal defence + green: ₹280 Cr<br>• <em>Total: ~₹790 Cr</em>`
            }
        };
        const cityDesc = descriptions[activeCity] || descriptions.lucknow;
        document.getElementById("projection-desc-notes").innerHTML = cityDesc[scenario] || cityDesc["2026"];
    }

    climateRadios.forEach(radio => {
        radio.addEventListener("change", (e) => {
            const val = e.target.value;
            updateProjectionDescription(val);
            
            if (val === "2026") {
                // Restore baseline
                resetGridState();
            } else {
                // Fetch projection for year 2050
                const rcp = val.split("-")[1]; // rcp45 or rcp85
                
                fetch("/api/future-projection", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        city: activeCity,
                        rcp: rcp 
                    })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success") {
                        // Re-map the projected data fields so they fit grid values
                        currentGrid = data.projection.map(proj => {
                            return {
                                zone_id: proj.zone_id,
                                name: proj.name,
                                x: proj.x,
                                y: proj.y,
                                latitude: proj.latitude,
                                longitude: proj.longitude,
                                lst_day_actual: proj.projected_temp, // map to LST active display properties
                                lst_day_pred: proj.projected_temp,
                                lst_night_actual: proj.projected_night_temp,
                                lst_night_pred: proj.projected_night_temp,
                                ndvi: proj.projected_ndvi,
                                albedo: proj.original_albedo, // fallback
                                isf: proj.projected_isf,
                                hvi: proj.original_temp / 50.0, // scale index heuristic for 2050 visualization
                                hvi_class: "Extreme Risk (2050 Projection)",
                                pop_density: proj.original_temp * 400.0, // fallback scaling
                                wind_speed: proj.projected_temp * 0.08,
                                distance_to_river: 5.0
                            };
                        });
                        
                        updateGridMaps();
                        renderGrid();
                        
                        if (selectedZone) {
                            selectZone(selectedZone.zone_id);
                        }
                    }
                })
                .catch(err => console.error("Error loading climate projections:", err));
            }
        });
    });

    // -------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------
    function formatCurrency(val) {
        if (val >= 10000000) {
            return `₹${(val / 10000000).toFixed(2)} Cr`;
        }
        if (val >= 100000) {
            return `₹${(val / 100000).toFixed(2)} L`;
        }
        return `₹${val.toLocaleString("en-IN")}`;
    }
});
