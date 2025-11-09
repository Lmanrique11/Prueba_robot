import uproot
import awkward as ak
import numpy as np
import pandas as pd
import wget
import os
import json
import shutil
from collections import defaultdict

# --- Constantes ---
MEV_TO_GEV = 1 / 1000.0

# === Funciones de Carga de Datos ===

def load_settings(settings_file="settings.json"):
    """Carga la configuración desde el archivo JSON."""
    print(f"Loading settings from {settings_file}...")
    with open(settings_file, 'r') as f:
        return json.load(f)

def download_data(urls):
    """Descarga los archivos .root si no existen localmente."""
    local_files = []
    os.makedirs("data", exist_ok=True)
    for url in urls:
        filename = os.path.basename(url)
        local_path = os.path.join("data", filename)
        if not os.path.exists(local_path):
            print(f"Downloading {url}...")
            wget.download(url, out=local_path)
        else:
            print(f"Found cached file: {local_path}")
        local_files.append(local_path)
    return local_files

def load_all_data(files, treename, branches):
    """Carga y concatena todos los archivos ROOT en un solo Awkward Array."""
    print("Loading and concatenating all data files...")
    all_events = uproot.concatenate(
        files,
        filter_name=branches,
        library="ak"
    )
    return all_events

# === Funciones de Física (Cortes y Cálculos) ===

def apply_base_cuts(events, cuts):
    """Aplica los cortes de calidad y preselección."""
    print("Applying base cuts...")
    
    # 1. Trigger
    events = events[events.trigP]
    
    # 2. Exactamente 2 fotones
    events = events[events.photon_n == 2]

    # 3. Cortes de calidad (isTightID)
    events = events[ak.all(events.photon_isTightID, axis=1)]

    # 4. Cortes de pT (en GeV)
    pt_min_mev = cuts["photon_pt_min_gev"] * 1000.0
    events = events[ak.all(events.photon_pt > pt_min_mev, axis=1)]

    # 5. Cortes de pseudorapidez (Eta)
    eta_max = cuts["photon_eta_max"]
    gap_low = cuts["photon_eta_gap_low"]
    gap_high = cuts["photon_eta_gap_high"]
    eta_mask = (abs(events.photon_eta) < eta_max) & \
               ((abs(events.photon_eta) < gap_low) | (abs(events.photon_eta) > gap_high))
    events = events[ak.all(eta_mask, axis=1)]

    # 6. Cortes de aislamiento (Isolation)
    ptcone30_ratio = events.photon_ptcone30 / events.photon_pt
    etcone20_ratio = events.photon_etcone20 / events.photon_pt
    iso_mask = (ptcone30_ratio < cuts["ptcone30_ratio_max"]) & \
               (etcone20_ratio < cuts["etcone20_ratio_max"])
    events = events[ak.all(iso_mask, axis=1)]
    
    print(f"Base cuts applied. {len(events)} events remaining.")
    return events

def calculate_kinematics(photons):
    """
    Calcula todas las variables cinemáticas para un conjunto de fotones.
    Devuelve un diccionario de arrays.
    """
    # Convertir a GeV en el momento del cálculo
    pt = photons.pt * MEV_TO_GEV
    E = photons.E * MEV_TO_GEV
    eta = photons.eta
    phi = photons.phi

    # Calcular componentes (como en tu imagen)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    # Masa (debería ser ~0 para fotones, pero es bueno tenerla)
    mass_sq = np.maximum(0, E**2 - (px**2 + py**2 + pz**2))
    m = np.sqrt(mass_sq)

    return {
        "px": px, "py": py, "pz": pz, "E": E,
        "pt": pt, "eta": eta, "phi": phi, "m": m
    }

def calculate_system_kinematics(ph1, ph2):
    """
    Calcula las variables cinemáticas para el sistema de dos fotones (di-fotón).
    ph1 y ph2 son los diccionarios de la función calculate_kinematics.
    """
    # Suma de 4-vectores
    E_sys = ph1["E"] + ph2["E"]
    px_sys = ph1["px"] + ph2["px"]
    py_sys = ph1["py"] + ph2["py"]
    pz_sys = ph1["pz"] + ph2["pz"]

    # Calcular variables del sistema (como en tu imagen)
    pt_sys = np.sqrt(px_sys**2 + py_sys**2)
    p_sys_sq = px_sys**2 + py_sys**2 + pz_sys**2
    
    # Masa Invariante (la variable más importante)
    mass_sq = np.maximum(0, E_sys**2 - p_sys_sq)
    m_sys = np.sqrt(mass_sq)

    # Pseudorapidez del sistema
    p_sys = np.sqrt(p_sys_sq)
    eta_sys = 0.5 * np.log((p_sys + pz_sys) / (p_sys - pz_sys))
    eta_sys = np.nan_to_num(eta_sys, nan=0.0, posinf=0.0, neginf=0.0) # Manejar división por cero

    # Ángulo azimutal del sistema
    phi_sys = np.arctan2(py_sys, px_sys)

    return {
        "px": px_sys, "py": py_sys, "pz": pz_sys, "E": E_sys,
        "pt": pt_sys, "eta": eta_sys, "phi": phi_sys, "m": m_sys
    }

def generate_cut_list(analysis_cuts):
    """Expande la configuración de 'analysis_cuts' a una lista de cortes específicos."""
    final_cuts = []
    for cut in analysis_cuts:
        if cut.get("type") == "scan":
            # Genera cortes para el scan
            for pt_min in np.arange(cut["pt_min_gev"], cut["pt_max_gev"], cut["pt_step_gev"]):
                final_cuts.append({
                    "name": f"scan_pt_{pt_min:.0f}GeV",
                    "pt_photon1_min_gev": pt_min,
                    "pt_photon2_min_gev": pt_min
                })
        else:
            # Añade el corte manual (simétrico o asimétrico)
            final_cuts.append(cut)
    return final_cuts

def get_statistics(data_dict):
    """Calcula estadísticas descriptivas para un diccionario de arrays."""
    stats = {}
    for name, array in data_dict.items():
        s = pd.Series(array)
        stats[name] = s.describe().to_dict()
        # pandas.describe() no es compatible con JSON, convertir tipos
        for key, val in stats[name].items():
            stats[name][key] = np.generic.item(val)
    return stats

# === Función Principal ===

def main():
    """Ejecuta el pipeline completo de análisis."""
    
    # 1. Cargar configuración
    settings = load_settings()
    io_cfg = settings["io"]
    phys_cfg = settings["physics"]
    
    # Limpiar/crear directorio de salida
    out_dir = io_cfg["output_dir"]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # 2. Cargar datos (descargar si es necesario)
    local_files = download_data(io_cfg["data_urls"])
    all_events = load_all_data(local_files, io_cfg["treename"], phys_cfg["branches"])

    # 3. Aplicar cortes base
    # (Ordenamos fotones por pT para que el [:, 0] sea siempre el líder)
    sorted_indices = ak.argsort(all_events.photon.pt, axis=1, ascending=False)
    all_events.photon = all_events.photon[sorted_indices]
    clean_events = apply_base_cuts(all_events, phys_cfg["base_cuts"])
    
    # 4. Calcular cinemática de TODOS los eventos "limpios" de una sola vez
    ph1_kin = calculate_kinematics(clean_events.photon[:, 0])
    ph2_kin = calculate_kinematics(clean_events.photon[:, 1])
    sys_kin = calculate_system_kinematics(ph1_kin, ph2_kin)
    
    # 5. Generar lista de todos los análisis a ejecutar
    analysis_jobs = generate_cut_list(phys_cfg["analysis_cuts"])
    print(f"\nGenerated {len(analysis_jobs)} analysis jobs from settings.")

    # 6. Bucle de Análisis: aplicar cortes de pT y calcular estadísticas
    all_results = {}
    
    for job in analysis_jobs:
        job_name = job["name"]
        print(f"--- Running job: {job_name} ---")
        
        # Crear máscara de corte de pT
        pt1_min = job["pt_photon1_min_gev"]
        pt2_min = job["pt_photon2_min_gev"]
        
        cut_mask = (ph1_kin["pt"] > pt1_min) & (ph2_kin["pt"] > pt2_min)
        
        if not ak.any(cut_mask):
            print(f"No events passed cut for {job_name}. Skipping.")
            continue
            
        # Preparar un diccionario con TODOS los arrays de variables
        all_variables = {}
        for (prefix, kin_dict) in [("ph1", ph1_kin), ("ph2", ph2_kin), ("sys", sys_kin)]:
            for (var, arr) in kin_dict.items():
                all_variables[f"{prefix}_{var}"] = arr[cut_mask]
        
        # Calcular estadísticas
        stats = get_statistics(all_variables)
        all_results[job_name] = stats
        
        # Guardar estadísticas en un archivo .js (como en tu script original)
        job_dir = os.path.join(out_dir, job_name)
        os.makedirs(job_dir, exist_ok=True)
        js_path = os.path.join(job_dir, "stats.js")
        
        # Formatear el nombre de la variable JS
        js_var_name = f"stats_{job_name.replace('-', '_')}"
        
        with open(js_path, 'w') as f:
            f.write(f"const {js_var_name} = ")
            json.dump(stats, f, indent=2)
            f.write(";\n")
            
        print(f"✅ Saved stats to {js_path}")

    # 7. Guardar un resumen de todos los resultados
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 Analysis complete! All results saved in '{out_dir}'.")


if __name__ == "__main__":
    main()
