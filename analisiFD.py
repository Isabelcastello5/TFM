# ================================================
# ANÀLISI DE DIMENSIÓ FRACTAL 
# ================================================
# Aquest script processa imatges de tomografia computada (format BMP) de
# mostres cranials per analitzar-ne les propietats mecàniques
# El flux de treball inclou:
# 1. Detecció automàtica del tall central i els talls veïns.
# 2. Selecció manual d'una ROI (regió d'interès) en l'slice central.
# 3. Extracció de la potència espectral (PSD) via transformada de Fourier.
# 4. Càlcul de la FD a partir de l'ajust lineal en escala log-log.
# 5. Exportació de resultats i gràfics en un directori de sortida.
# Aquest codi és útil per estudis biomecànics que correlacionin la microestructura òssia
# amb propietats mecàniques.
# ================================================

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft2, fftshift
from matplotlib.widgets import Cursor



# === CONFIGURACIÓ DE DIRECTORIS ===

#03B
bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/03B/03B_LR_center"
output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/03B/PROVA"

#03D
#bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/03D/2934D_24_LR_center"
#output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/03D/PROVA2"

#04A
#bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/04A/04A_mCT_center"
#output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/04A/PROVA2"

#06A
#bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/06A/06A_LR_center"
#output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/06A/PROVA2"

#06B
#bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/06B/06B_LR_center"
#output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/06B/PROVA2"

#07A
#bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/07A/07A_LR_center"
#output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/07A/PROVA2"

#07B
#bmp_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/07B/07B_LR_center"
#output_folder = "/Users/isabelcastelloortega/Desktop/TFM/CRANI/07B/PROVA2"


os.makedirs(output_folder, exist_ok=True)

# PARÀMETRES D'ANÀLISI
res_mm = 0.038
offsets_mm = [-2, -1, 0, 1, 2]
pixel_size_mm = 0.038
crop_size = 200

# FUNCIONS AUXILIARS
def get_radial_profile(psd2d):
    """
    Calcula el perfil radial (1D) d'una imatge PSD (2D), mitjançant mitjanes
    concèntriques des del centre. Això permet simplificar l'anàlisi espectral.

    Paràmetres:
        psd2d: Imatge 2D de l'espectre de potència (Power Spectral Density)

    Retorna:
        Vector 1D amb la mitjana de potència per cada radi (perfil radial)
    """
        
    center = np.array(psd2d.shape) // 2
    y, x = np.indices(psd2d.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / (nr + 1e-8)

def compute_fd(img, res_mm, start_manual=5, end_manual=49):
    """
    Calcula la dimensió fractal (FD) d'una ROI mitjançant el mètode de PSD radial
    i regressió log-log sobre un interval seleccionat manualment.

    Paràmetres:
        img: Imatge 2D (ROI) en escala de grisos
        res_mm: Resolució espacial en mm/píxel
        start_manual: Índex inicial dels punts a utilitzar per ajustar
        end_manual: Índex final dels punts a descartar per ajustar

    Retorna:
        fd: dimensió fractal estimada
        slope: pendent de l'ajust log-log
        intercept: tall amb l'eix Y
        r_squared: coeficient de determinació de l'ajust
        (lambda_vals, psd_total): vectors complets
        (lambda_used, psd_used): vectors usats per ajustar
        mask_used: màscara booleana indicant els punts usats
    """
    img_float = (img.astype(np.float32) - img.min()) / (img.max() - img.min() + 1e-10)
    fft_img = (np.abs(fftshift(fft2(img_float)))**2) / (img.shape[0] * img.shape[1])
    psd_full = get_radial_profile(fft_img)
    N = img.shape[0]
    r = np.arange(len(psd_full))
    freqs = r / (N * res_mm)
    lambda_vals = 1 / freqs[1:]
    psd = psd_full[1:len(lambda_vals)+1]
    mask_valid = ~np.isnan(psd)
    psd_total = psd[mask_valid]
    lambda_total = lambda_vals
    total_points = len(psd_total)
    if total_points <= start_manual + end_manual:
        return None
    psd_used = psd_total[start_manual:total_points - end_manual]
    lambda_used = lambda_total[:len(psd_used)]
    if len(psd_used) < 5:
        return None
    log_lambda = np.log10(lambda_used + 1e-10)
    log_psd = np.log10(psd_used + 1e-10)
    slope, intercept = np.linalg.lstsq(np.vstack([log_lambda, np.ones_like(log_lambda)]).T, log_psd, rcond=None)[0]
    y_pred = slope * log_lambda + intercept
    r_squared = 1 - np.sum((log_psd - y_pred)**2) / np.sum((log_psd - np.mean(log_psd))**2)
    fd = 4 - slope / 2
    mask_used = np.zeros_like(psd_total, dtype=bool)
    mask_used[start_manual:total_points - end_manual] = True
    return fd, slope, intercept, r_squared, (lambda_vals, psd_total), (lambda_used, psd_used), mask_used

# DETECCIÓ DE FITXERS I SLICES
bmp_files = sorted([f for f in os.listdir(bmp_folder) if f.endswith('.bmp')])
if not bmp_files:
    raise ValueError(f"No s'han trobat fitxers .bmp a: {bmp_folder}")
match = re.match(r"(.+?)(\d+)\.bmp", bmp_files[0])
if not match:
    raise ValueError(f"No s'ha pogut detectar el prefix: {bmp_files[0]}")
prefix = match.group(1)
num_digits = len(match.group(2))
slice_numbers = sorted([int(re.search(r'(\d{4,})\.bmp$', f).group(1)) for f in bmp_files])
centre = slice_numbers[len(slice_numbers) // 2]

# PROCÉS PRINCIPAL
resultats = []
aout_all_df = None
coords_guardades = None
for offset_mm in offsets_mm:
    slice_id = int(round(centre + offset_mm / res_mm))
    filename = f"{prefix}{slice_id:0{num_digits}d}.bmp"
    path = os.path.join(bmp_folder, filename)
    if not os.path.exists(path):
        print(f"⚠️ No trobat: {filename}")
        continue
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if coords_guardades is None:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
        coords = []
        def onclick(event):
            coords.append((int(event.xdata), int(event.ydata)))
            plt.close()
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.title("Fes clic al centre de la indentació")
        plt.show()
        if not coords:
            raise ValueError("No s'ha seleccionat cap punt")
        coords_guardades = coords[0]
    x, y = coords_guardades
    half = crop_size // 2
    crop = img[y - half:y + half, x - half:x + half]
    cv2.imwrite(os.path.join(output_folder, f"crop_{slice_id}.bmp"), crop)

    densitat_relativa = np.mean(crop)
    fd_list, r2_list = [], []
    for rep in range(5):
        res = compute_fd(crop, pixel_size_mm, start_manual=3, end_manual=26)
        if res is None:
            continue
        fd, slope, intercept, r_squared, (lambda_total, psd_total), (lambda_used, psd_used), mask_used = res
        fd_list.append(fd)
        r2_list.append(r_squared)
        name = f"{slice_id}_rep{rep+1}"
        psd_useds_masked = np.full_like(psd_total, np.nan)
        psd_useds_masked[mask_used] = psd_total[mask_used]
        psd_descartats_masked = np.full_like(psd_total, np.nan)
        psd_descartats_masked[~mask_used] = psd_total[~mask_used]
        plt.figure()
        plt.loglog(lambda_total, psd_total, 'b-', label='PSD completa')
        plt.plot(lambda_total[mask_used], psd_total[mask_used], 'ro', label='useds')
        plt.plot(lambda_total[~mask_used], psd_total[~mask_used], 'o', color='gray', label='Descartats')
        ajuste = 10**(slope * np.log10(lambda_total[mask_used] + 1e-10) + intercept)
        plt.plot(lambda_total[mask_used], ajuste, 'r--', label=f'Ajust lineal')
        plt.xlabel("Longitud d'ona λ (µm)")
        plt.ylabel("Espectre de potència (PSD)")
        plt.title(f"Slice {slice_id} - Rep {rep+1}\nPendent = {slope:.3f}, R² = {r_squared:.3f}")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"loglog_{name}.png"))
        plt.close()
        if rep == 0:
            aout_df = pd.DataFrame({
                'lambda': lambda_total,
                f'PSD_{slice_id}': psd_total,
                f'PSD_{slice_id}_usado': psd_useds_masked
            })
            aout_df.to_excel(os.path.join(output_folder, f"Aout_{slice_id}.xlsx"), index=False)
            if aout_all_df is None:
                aout_all_df = aout_df.copy()
            else:
                aout_all_df = pd.merge(aout_all_df, aout_df, on='lambda', how='outer')
    if fd_list:
        resultats.append({
            'slice': slice_id,
            'FD_promig': np.mean(fd_list),
            'slope': slope,
            'R2_promig': np.mean(r2_list)
            'Intensitat_mitjana': densitat_relativa
        })
        print(f"Slice {slice_id}: FD = {np.mean(fd_list):.4f}, R² = {np.mean(r2_list):.4f}")

# GUARDAR RESULTATS 
if aout_all_df is not None:
    aout_all_df = aout_all_df.sort_values(by='lambda', ascending=False)
    aout_all_df.to_excel(os.path.join(output_folder, "Aout_total.xlsx"), index=False)
pd.DataFrame(resultats).to_excel(os.path.join(output_folder, "FD_resultats.xlsx"), index=False)
