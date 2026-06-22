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
    
    const citySelectEl = document.getElementById("city-select");
    
    // UI Selectors
    const cityGridEl = document.getElementById("city-grid");
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
    
    // Scenario Simulator Selectors
    const slideTree = document.getElementById("slide-tree");
    const slideRoofs = document.getElementById("slide-roofs");
    const slidePavement = document.getElementById("slide-pavement");
    const valSlideTree = document.getElementById("val-slide-tree");
    const valSlideRoofs = document.getElementById("val-slide-roofs");
    const valSlidePavement = document.getElementById("val-slide-pavement");
    const btnRunSim = document.getElementById("btn-run-simulation");
    const btnResetSim = document.getElementById("btn-reset-simulation");
    const badgeCooling = document.getElementById("sim-cooling-badge");
    const valSimDelta = document.getElementById("val-sim-delta");
    const badgeCost = document.getElementById("sim-cost-badge");
    const valSimCost = document.getElementById("val-sim-cost");
    
    // Optimizer Selectors
    const optBudgetSlider = document.getElementById("opt-budget");
    const valOptBudget = document.getElementById("val-opt-budget");
    const optHviWeight = document.getElementById("opt-hvi-weight");
    const btnRunOpt = document.getElementById("btn-run-optimizer");
    const btnResetOpt = document.getElementById("btn-reset-optimizer");
    const badgeOptSummary = document.getElementById("opt-summary-stats");
    const valOptSpent = document.getElementById("val-opt-spent");
    const valOptCount = document.getElementById("val-opt-count");
    const optTableBody = document.querySelector("#opt-table tbody");
    
    // Climate Projection Selectors
    const climateRadios = document.querySelectorAll("input[name='climate-year']");
    
    // Initialize App Data
    fetchCityData();

    // -------------------------------------------------------------
    // 1. Data Fetching
    // -------------------------------------------------------------
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
                    document.getElementById("opt-table-header").innerText = data.city + " Locality";
                    document.getElementById("projection-city-title").innerText = data.city + " 2050 Projection Engine";
                    
                    let subNotes = "";
                    if (activeCity === "lucknow") subNotes = "(Jankipuram, Vrindavan Yojana)";
                    else if (activeCity === "delhi") subNotes = "(Dwarka, Okhla)";
                    else if (activeCity === "kanpur") subNotes = "(Kalyanpur, Jajmau)";
                    else if (activeCity === "goa") subNotes = "(Vasco Port, Miramar)";
                    else if (activeCity === "mumbai") subNotes = "(Bandra, Juhu)";
                    
                    document.getElementById("projection-desc-notes").innerHTML = 
                        `<strong>Urban Expansion Model:</strong> Projects 25% citywide population surge, and a 15% increase in impervious surfaces (loss of vegetation) in peripheral suburban blocks ${subNotes}.`;
                    
                    // Reset local overrides
                    activeInterventions = {};
                    currentGrid.forEach(z => {
                        activeInterventions[z.zone_id] = {
                            tree_planting: 0,
                            green_roofs: 0,
                            cool_pavement: 0
                        };
                    });
                    
                    renderGrid();
                    updateLegend();
                    setupVectorLayers();
                } else {
                    cityGridEl.innerHTML = `<div class="grid-loading text-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(err => {
                console.error("Fetch error:", err);
                cityGridEl.innerHTML = `<div class="grid-loading text-danger">Server Offline</div>`;
            });
    }
    
    // Hook up city select change listener
    if (citySelectEl) {
        citySelectEl.addEventListener("change", (e) => {
            activeCity = e.target.value;
            selectedZone = null;
            detailsPlaceholder.classList.remove("hidden");
            detailsActive.classList.add("hidden");
            // Reset sliders panel state
            slideTree.disabled = true;
            slideRoofs.disabled = true;
            slidePavement.disabled = true;
            btnRunSim.disabled = true;
            badgeCooling.classList.add("hidden");
            badgeCost.classList.add("hidden");
            
            // Reset projections to baseline
            document.querySelector("input[name='climate-year'][value='2026']").checked = true;
            fetchCityData();
        });
    }

    // -------------------------------------------------------------
    // 2. Map Rendering (Grid Map)
    // -------------------------------------------------------------
    function renderGrid() {
        cityGridEl.innerHTML = "";
        
        currentGrid.forEach(zone => {
            const cell = document.createElement("div");
            cell.className = "grid-cell";
            cell.id = `cell-${zone.zone_id}`;
            
            // Set cell colors based on the current metric view
            const val = getMetricValue(zone, currentView);
            let color = getColorForValue(val, currentView, zone);
            
            // Direct sea blue coloring for water body cells
            if (zone.is_water) {
                color = "#004b6b";
            }
            cell.style.backgroundColor = color;
            
            // Mark cells near Gomti River/Yamuna for border formatting
            if (zone.distance_to_river < 1.1 && !zone.is_water) {
                cell.classList.add("near-river");
            }
            
            // Highlight wind pathways if active
            const isCorridorActive = toggleCorridorsCheckbox.checked;
            if (isCorridorActive && isWindCorridorZone(zone.zone_id)) {
                cell.classList.add("wind-path");
                cell.style.boxShadow = "inset 0 0 10px rgba(0, 245, 212, 0.4)";
            }
            
            // Selection formatting
            if (selectedZone && selectedZone.zone_id === zone.zone_id) {
                cell.classList.add("selected");
            }
            
            // Tooltip / Hover label
            if (zone.is_water) {
                cell.title = `${zone.name}\nLST: ${getLSTValue(zone).toFixed(1)}°C\nWater Body - Interventions Blocked`;
            } else {
                cell.title = `${zone.name}\nLST (Day): ${getLSTValue(zone).toFixed(1)}°C\nHVI: ${zone.hvi.toFixed(2)} (${zone.hvi_class})`;
            }
            
            // Click Handler
            cell.addEventListener("click", () => selectZone(zone.zone_id));
            
            cityGridEl.appendChild(cell);
        });
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
            // Scale between 35°C (Cool/Green) and 45°C (Extreme Heat/Red)
            const minT = 34.0;
            const maxT = 45.0;
            const norm = Math.max(0, Math.min(1, (val - minT) / (maxT - minT)));
            // HSL path: Green (120) -> Yellow (60) -> Red (0)
            const hue = (1.0 - norm) * 120;
            return `hsl(${hue}, 80%, 45%)`;
        } 
        else if (view === "lst_night") {
            // Scale between 26°C and 34°C
            const minT = 26.0;
            const maxT = 34.0;
            const norm = Math.max(0, Math.min(1, (val - minT) / (maxT - minT)));
            const hue = (1.0 - norm) * 120;
            return `hsl(${hue}, 80%, 45%)`;
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
    function setupVectorLayers() {
        // Remove old overlays if any
        const oldOverlays = document.querySelectorAll(".river-svg-overlay, .corridor-svg-overlay");
        oldOverlays.forEach(el => el.remove());
        
        const wrapper = document.querySelector(".map-wrapper");
        const mapW = 600;
        const mapH = 600;
        const cellS = mapW / 15;
        
        // Generate path using city centerline coordinates
        let pathD = "";
        if (cityData.water_type === "river_gomti") {
            // Gomti River
            for (let row = 0; row < 15; row += 0.5) {
                const yr = row;
                const xr = 13.0 - 0.8 * yr - 2.0 * Math.sin(yr / 2.0);
                const px = (xr + 0.5) * cellS;
                const py = (yr + 0.5) * cellS;
                if (row === 0) pathD += `M ${px} ${py}`;
                else pathD += ` L ${px} ${py}`;
            }
        } else if (cityData.water_type === "river_yamuna") {
            // Yamuna River in Delhi
            for (let row = 0; row < 15; row += 0.5) {
                const yr = row;
                const xr = 9.0 - 0.3 * yr - 1.5 * Math.cos(yr / 3.0);
                const px = (xr + 0.5) * cellS;
                const py = (yr + 0.5) * cellS;
                if (row === 0) pathD += `M ${px} ${py}`;
                else pathD += ` L ${px} ${py}`;
            }
        } else if (cityData.water_type === "river_ganges") {
            // Ganges River in Kanpur
            for (let col = 0; col < 15; col += 0.5) {
                const xr = col;
                const yr = 13.0 + 0.5 * Math.sin(xr / 2.0);
                const px = (xr + 0.5) * cellS;
                const py = (yr + 0.5) * cellS;
                if (col === 0) pathD += `M ${px} ${py}`;
                else pathD += ` L ${px} ${py}`;
            }
        }
        
        // --- 1. Water River SVG ---
        const riverSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        riverSvg.setAttribute("class", "river-svg-overlay");
        riverSvg.setAttribute("viewBox", `0 0 ${mapW} ${mapH}`);
        
        if (pathD !== "") {
            const riverPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
            riverPath.setAttribute("class", "river-path");
            riverPath.setAttribute("d", pathD);
            riverSvg.appendChild(riverPath);
        }
        wrapper.appendChild(riverSvg);
        
        // --- 2. Wind Corridor SVG ---
        const corridorSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        corridorSvg.setAttribute("class", "corridor-svg-overlay");
        corridorSvg.setAttribute("viewBox", `0 0 ${mapW} ${mapH}`);
        
        // Render arrows or pathways based on wind corridors
        let pathD1 = "";
        let pathD2 = "";
        
        if (cityData.water_type.startsWith("river")) {
            pathD1 = `M ${1 * cellS} ${3 * cellS} C ${4 * cellS} ${5 * cellS}, ${8 * cellS} ${8 * cellS}, ${13 * cellS} ${13 * cellS}`;
            pathD2 = `M ${10 * cellS} ${1 * cellS} L ${12 * cellS} ${8 * cellS} L ${13 * cellS} ${14 * cellS}`;
        } else {
            // Coastal cities flow from west to east
            pathD1 = `M ${0.5 * cellS} ${4 * cellS} L ${14 * cellS} ${4 * cellS}`;
            pathD2 = `M ${0.5 * cellS} ${10 * cellS} L ${14 * cellS} ${10 * cellS}`;
        }
        
        const windLine1 = document.createElementNS("http://www.w3.org/2000/svg", "path");
        windLine1.setAttribute("class", "corridor-line");
        windLine1.setAttribute("d", pathD1);
        corridorSvg.appendChild(windLine1);
        
        const windLine2 = document.createElementNS("http://www.w3.org/2000/svg", "path");
        windLine2.setAttribute("class", "corridor-line");
        windLine2.setAttribute("d", pathD2);
        corridorSvg.appendChild(windLine2);
        
        wrapper.appendChild(corridorSvg);
        
        // Bind visibility based on state
        updateVectorVisibility();
    }

    function updateVectorVisibility() {
        const riverSvg = document.querySelector(".river-svg-overlay");
        const corridorSvg = document.querySelector(".corridor-svg-overlay");
        
        if (riverSvg) {
            if (toggleRiverCheckbox.checked) {
                riverSvg.classList.remove("hidden");
            } else {
                riverSvg.classList.add("hidden");
            }
        }
        
        if (corridorSvg) {
            if (toggleCorridorsCheckbox.checked) {
                corridorSvg.classList.add("visible");
            } else {
                corridorSvg.classList.remove("visible");
            }
        }
    }

    toggleRiverCheckbox.addEventListener("change", updateVectorVisibility);
    toggleCorridorsCheckbox.addEventListener("change", () => {
        updateVectorVisibility();
        renderGrid(); // Redraw grid cells to apply animated wind classes
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
            const minT = currentView === "lst_day" ? "34°C" : "26°C";
            const maxT = currentView === "lst_day" ? "45°C+" : "34°C+";
            legendEl.innerHTML = `
                <div class="legend-item"><span class="legend-color" style="background: hsl(120, 80%, 45%)"></span> <span>Cool (${minT})</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(60, 80%, 45%)"></span> <span>Moderate</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(30, 80%, 45%)"></span> <span>Warm</span></div>
                <div class="legend-item"><span class="legend-color" style="background: hsl(0, 80%, 45%)"></span> <span>Extreme Heat (${maxT})</span></div>
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
        selectedZone = currentGrid.find(z => z.zone_id === zoneId);
        
        // Highlight active cell on map
        document.querySelectorAll(".grid-cell").forEach(el => el.classList.remove("selected"));
        const cell = document.getElementById(`cell-${zoneId}`);
        if (cell) cell.classList.add("selected");
        
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
        
        // Handle water zones simulation bypass
        if (selectedZone.is_water) {
            slideTree.disabled = true;
            slideRoofs.disabled = true;
            slidePavement.disabled = true;
            btnRunSim.disabled = true;
            btnRunSim.innerText = "Intervention Blocked";
            badgeCooling.classList.add("hidden");
            badgeCost.classList.add("hidden");
        } else {
            btnRunSim.innerText = "Simulate Local Coolings";
            enableSimulationSliders();
        }
        
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
    // 7. Local Scenario Simulation
    // -------------------------------------------------------------
    function enableSimulationSliders() {
        slideTree.disabled = false;
        slideRoofs.disabled = false;
        slidePavement.disabled = false;
        
        // Load current simulation levels
        const simState = activeInterventions[selectedZone.zone_id];
        slideTree.value = simState.tree_planting;
        slideRoofs.value = simState.green_roofs;
        slidePavement.value = simState.cool_pavement;
        
        updateSliderLabels();
        
        btnRunSim.disabled = false;
        
        // Check if values are zero
        if (simState.tree_planting > 0 || simState.green_roofs > 0 || simState.cool_pavement > 0) {
            // calculate the cooling result displays
            calculateLocalCoolingDisplay();
        } else {
            badgeCooling.classList.add("hidden");
            badgeCost.classList.add("hidden");
        }
    }

    function updateSliderLabels() {
        valSlideTree.innerText = `${slideTree.value}%`;
        valSlideRoofs.innerText = `${slideRoofs.value}%`;
        valSlidePavement.innerText = `${slidePavement.value}%`;
    }

    [slideTree, slideRoofs, slidePavement].forEach(slider => {
        slider.addEventListener("input", () => {
            updateSliderLabels();
            btnRunSim.disabled = false;
        });
    });

    btnRunSim.addEventListener("click", () => {
        if (!selectedZone) return;
        
        const zoneId = selectedZone.zone_id;
        
        // Record interventions in local state
        activeInterventions[zoneId] = {
            tree_planting: parseInt(slideTree.value),
            green_roofs: parseInt(slideRoofs.value),
            cool_pavement: parseInt(slidePavement.value)
        };
        
        // Call server to calculate simulation
        btnRunSim.disabled = true;
        btnRunSim.innerText = "Simulating...";
        
        fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                city: activeCity,
                zone_interventions: activeInterventions 
            })
        })
        .then(res => res.json())
        .then(data => {
            btnRunSim.innerText = "Simulate Local Coolings";
            btnRunSim.disabled = false;
            
            if (data.status === "success") {
                currentGrid = data.full_grid;
                
                // Redraw map with updated temperatures
                renderGrid();
                
                // Highlight select
                const cell = document.getElementById(`cell-${zoneId}`);
                if (cell) cell.classList.add("selected");
                
                // Get updated zone object
                selectedZone = currentGrid.find(z => z.zone_id === zoneId);
                
                // Update results displays
                const simZoneRes = data.simulated_zones.find(z => z.zone_id === zoneId);
                if (simZoneRes) {
                    valLstDay.innerText = `${selectedZone.lst_day_pred.toFixed(1)}°C`;
                    valLstNight.innerText = `${selectedZone.lst_night_pred.toFixed(1)}°C`;
                    valNdvi.innerText = selectedZone.ndvi.toFixed(2);
                    valAlbedo.innerText = selectedZone.albedo.toFixed(2);
                    valIsf.innerText = `${Math.round(selectedZone.isf * 100)}%`;
                    
                    badgeCooling.classList.remove("hidden");
                    valSimDelta.innerText = `-${simZoneRes.temp_reduction.toFixed(2)}°C`;
                    
                    badgeCost.classList.remove("hidden");
                    valSimCost.innerText = formatCurrency(simZoneRes.interventions.tree_planting * 80000 + 
                                                         simZoneRes.interventions.green_roofs * 150000 + 
                                                         simZoneRes.interventions.cool_pavement * 100000);
                }
                
                // Update SHAP values
                fetchAndDrawShap(zoneId);
            }
        })
        .catch(err => {
            console.error("Simulation error:", err);
            btnRunSim.innerText = "Simulate Local Coolings";
            btnRunSim.disabled = false;
        });
    });

    btnResetSim.addEventListener("click", () => {
        if (!selectedZone) return;
        const zoneId = selectedZone.zone_id;
        
        slideTree.value = 0;
        slideRoofs.value = 0;
        slidePavement.value = 0;
        updateSliderLabels();
        
        activeInterventions[zoneId] = {
            tree_planting: 0,
            green_roofs: 0,
            cool_pavement: 0
        };
        
        btnRunSim.click(); // Re-trigger simulation with zeroes to reset
    });

    function calculateLocalCoolingDisplay() {
        const zoneId = selectedZone.zone_id;
        const baselineZone = cityData.zones.find(z => z.zone_id === zoneId);
        const cooling = baselineZone.lst_day_actual - selectedZone.lst_day_pred;
        
        if (cooling > 0.05) {
            badgeCooling.classList.remove("hidden");
            valSimDelta.innerText = `-${cooling.toFixed(2)}°C`;
            
            const cost = (
                activeInterventions[zoneId].tree_planting * 80000 +
                activeInterventions[zoneId].green_roofs * 150000 +
                activeInterventions[zoneId].cool_pavement * 100000
            );
            badgeCost.classList.remove("hidden");
            valSimCost.innerText = formatCurrency(cost);
        } else {
            badgeCooling.classList.add("hidden");
            badgeCost.classList.add("hidden");
        }
    }

    // -------------------------------------------------------------
    // 8. Municipal Budget Optimization Solver
    // -------------------------------------------------------------
    optBudgetSlider.addEventListener("input", () => {
        valOptBudget.innerText = formatCurrency(optBudgetSlider.value);
    });

    btnRunOpt.addEventListener("click", () => {
        const budget = optBudgetSlider.value;
        const weightByHvi = optHviWeight.checked;
        
        btnRunOpt.disabled = true;
        btnRunOpt.innerText = "Optimising...";
        
        fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                city: activeCity,
                budget: budget,
                weight_by_hvi: weightByHvi
            })
        })
        .then(res => res.json())
        .then(data => {
            btnRunOpt.innerText = "Run Budget Optimisation";
            btnRunOpt.disabled = false;
            
            if (data.status === "success") {
                currentGrid = data.full_grid;
                
                // Redraw map (automatically shows the cooled/optimized temperatures)
                renderGrid();
                
                // Show optimizer results
                badgeOptSummary.classList.remove("hidden");
                valOptSpent.innerText = formatCurrency(data.total_spent_inr);
                valOptCount.innerText = data.recommendations.length;
                
                // Render recommendations table
                renderOptimizerTable(data.recommendations);
                
                // Clear any local simulation slider overlays
                if (selectedZone) {
                    selectZone(selectedZone.zone_id);
                }
            }
        })
        .catch(err => {
            console.error("Optimization error:", err);
            btnRunOpt.innerText = "Run Budget Optimisation";
            btnRunOpt.disabled = false;
        });
    });

    function renderOptimizerTable(recs) {
        optTableBody.innerHTML = "";
        
        if (recs.length === 0) {
            optTableBody.innerHTML = `
                <tr class="empty-table-row">
                    <td colspan="7">No interventions could be allocated. Try increasing the capital budget.</td>
                </tr>
            `;
            return;
        }
        
        recs.forEach(rec => {
            const row = document.createElement("tr");
            
            // Map HVI value to badge styling
            let hviLabel = "Low";
            let hviColor = "text-success";
            if (rec.hvi >= 0.68) { hviLabel = "Extreme"; hviColor = "text-danger"; }
            else if (rec.hvi >= 0.52) { hviLabel = "High"; hviColor = "text-warning"; }
            else if (rec.hvi >= 0.35) { hviLabel = "Medium"; hviColor = "text-info"; }
            
            row.innerHTML = `
                <td><strong>${rec.name}</strong></td>
                <td class="${hviColor}"><strong>${hviLabel}</strong></td>
                <td>${rec.tree_planting > 0 ? rec.tree_planting + '%' : '-'}</td>
                <td>${rec.green_roofs > 0 ? rec.green_roofs + '%' : '-'}</td>
                <td>${rec.cool_pavement > 0 ? rec.cool_pavement + '%' : '-'}</td>
                <td><strong>${formatCurrency(rec.cost)}</strong></td>
                <td class="text-success font-weight-bold">-${rec.cooling_impact.toFixed(2)}°C</td>
            `;
            
            // Allow row clicking to select that zone on map
            row.style.cursor = "pointer";
            row.addEventListener("click", () => {
                selectZone(rec.zone_id);
            });
            
            optTableBody.appendChild(row);
        });
    }

    btnResetOpt.addEventListener("click", () => {
        // Clear optimization state and restore baseline grid
        if (cityData) {
            currentGrid = JSON.parse(JSON.stringify(cityData.zones));
            activeInterventions = {};
            currentGrid.forEach(z => {
                activeInterventions[z.zone_id] = { tree_planting: 0, green_roofs: 0, cool_pavement: 0 };
            });
            
            renderGrid();
            
            // Hide summary stats
            badgeOptSummary.classList.add("hidden");
            
            // Reset table
            optTableBody.innerHTML = `
                <tr class="empty-table-row">
                    <td colspan="7">Adjust capital slider and run optimizer to generate budget allocation.</td>
                </tr>
            `;
            
            // Reset detail view
            if (selectedZone) {
                selectZone(selectedZone.zone_id);
            }
        }
    });

    // -------------------------------------------------------------
    // 9. Climate Projections 2050
    // -------------------------------------------------------------
    climateRadios.forEach(radio => {
        radio.addEventListener("change", (e) => {
            const val = e.target.value;
            
            if (val === "2026") {
                // Restore baseline
                btnResetOpt.click();
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
