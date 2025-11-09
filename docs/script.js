// Archivos base que necesitamos
const summaryFile = 'summary.json';
const cutSelector = document.getElementById('cut-selector');
const statsContainer = document.getElementById('stats-table-container');
const plotsContainer = document.getElementById('plots-container');

// URL base de la página para construir las rutas de los archivos
const baseURL = window.location.pathname.endsWith('/') 
    ? window.location.pathname 
    : window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);

// ---------------------------------------------
// Función para cargar los resultados iniciales
// ---------------------------------------------
async function loadSummary() {
    try {
        // 1. Cargar el summary.json que tiene todos los resultados
        const response = await fetch(summaryFile);
        if (!response.ok) {
            throw new Error(`Error al cargar ${summaryFile}: ${response.statusText}`);
        }
        const summary = await response.json();
        const cutNames = Object.keys(summary).sort();

        // 2. Llenar el selector de cortes
        cutNames.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name.replace(/_/g, ' ');
            cutSelector.appendChild(option);
        });

        // 3. Manejar el cambio de selección
        cutSelector.addEventListener('change', (e) => loadCutResults(e.target.value));

        // 4. Cargar el primer corte por defecto
        if (cutNames.length > 0) {
            cutSelector.value = cutNames[0];
            loadCutResults(cutNames[0]);
        }

    } catch (error) {
        console.error("Error en la carga inicial:", error);
        statsContainer.innerHTML = `<p style="color: red;">Error: No se pudo cargar el archivo ${summaryFile}. Asegúrate de que el GitHub Action se ejecutó correctamente y generó la carpeta 'docs'.</p>`;
    }
}

// ---------------------------------------------
// Función para cargar un análisis de corte específico
// ---------------------------------------------
async function loadCutResults(cutName) {
    statsContainer.innerHTML = '<p>Cargando datos...</p>';
    plotsContainer.innerHTML = '';
    
    // La variable JS se llama stats_nombre_del_corte_sin_guiones
    const jsVarName = `stats_${cutName.replace(/-/g, '_')}`;
    const statsJSPath = `${cutName}/stats.js`;
    
    try {
        // 1. Cargar el archivo stats.js
        const script = document.createElement('script');
        script.src = statsJSPath;
        script.onload = () => {
            // El archivo stats.js cargado define una variable global (ej: const stats_sym_50GeV = {...})
            const statsData = window[jsVarName];
            
            if (statsData) {
                displayStats(statsData);
                displayPlots(cutName, statsData);
            } else {
                 statsContainer.innerHTML = `<p style="color: red;">Error: La variable ${jsVarName} no fue definida en ${statsJSPath}.</p>`;
            }
        };
        script.onerror = () => {
            statsContainer.innerHTML = `<p style="color: red;">Error al cargar el script ${statsJSPath}.</p>`;
        };
        document.head.appendChild(script);

    } catch (error) {
        console.error("Error al cargar los resultados del corte:", error);
    }
}

// ---------------------------------------------
// Función para mostrar la tabla de estadísticas
// ---------------------------------------------
function displayStats(statsData) {
    let html = '<table><thead><tr><th>Variable</th><th>Eventos (count)</th><th>Media (mean)</th><th>Std Dev (std)</th><th>Min</th><th>Max</th></tr></thead><tbody>';
    
    // Usamos las variables en orden lógico (pT, Masa, Eta, Phi)
    const displayOrder = ["pt", "m", "E", "eta", "phi"];
    const variableKeys = Object.keys(statsData).sort((a, b) => {
        // Ordena primero por prefix (sys, ph1, ph2) y luego por variable
        const aPrefix = a.split('_')[0];
        const bPrefix = b.split('_')[0];
        const aVar = a.split('_')[1];
        const bVar = b.split('_')[1];

        if (aPrefix !== bPrefix) {
            return aPrefix === "sys" ? -1 : (bPrefix === "sys" ? 1 : 0);
        }
        return displayOrder.indexOf(aVar) - displayOrder.indexOf(bVar);
    });

    variableKeys.forEach(key => {
        const stats = statsData[key];
        const cleanKey = key.replace("ph1", "Photon 1").replace("ph2", "Photon 2").replace("sys", "Sistema").replace("_", " ");
        
        html += `
            <tr>
                <td><strong>${cleanKey}</strong></td>
                <td>${stats.count.toFixed(0)}</td>
                <td>${stats.mean.toFixed(3)}</td>
                <td>${stats.std.toFixed(3)}</td>
                <td>${stats.min.toFixed(3)}</td>
                <td>${stats.max.toFixed(3)}</td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    statsContainer.innerHTML = html;
}

// ---------------------------------------------
// Función para mostrar los gráficos (iFrames)
// ---------------------------------------------
function displayPlots(cutName, statsData) {
    plotsContainer.innerHTML = ''; // Limpiar contenedores de plots
    
    const variableKeys = Object.keys(statsData);
    
    variableKeys.forEach(key => {
        // Construimos el nombre del archivo HTML que generó Plotly
        const filename = `${cutName}_${key}.html`.replace(/:/g, '').replace(/ /g, '_').replace(/\$/g, '').replace(/\\/g, '');
        const plotPath = `${cutName}/plots/${filename}`;
        
        // Crear el contenedor y el iFrame
        const iframe = document.createElement('iframe');
        iframe.className = 'plot-iframe';
        iframe.src = plotPath;
        iframe.title = key;
        
        plotsContainer.appendChild(iframe);
    });

    if (variableKeys.length === 0) {
        plotsContainer.innerHTML = '<p>No se encontraron variables para graficar.</p>';
    }
}


// Iniciar la aplicación
loadSummary();
