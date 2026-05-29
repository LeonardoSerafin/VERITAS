import io
from PIL import Image
import numpy as np
import cv2
from masfactory import ImageAsset

# Configurazioni delle soglie per le degradazioni
TRIGGER_CONFIG = {
    'exposure': {
        'dark_fixable_mean': 85, 'dark_bad_mean': 35,
        'bright_fixable_mean': 165, 'bright_bad_mean': 245,
        'dark_fixable_ratio': 0.08, 'dark_bad_ratio': 0.50,
        'bright_fixable_ratio': 0.05, 'bright_bad_ratio': 0.65,
    },
    'impulse_noise': {'fixable_impulse_ratio': 0.008, 'bad_impulse_ratio': 0.08},
    'jpeg': {'fixable_blockiness': 1.10, 'bad_blockiness': 1.80},
    'blur': {'fixable_laplacian_max': 500, 'bad_laplacian_max': 5},
}

def center_crop(image: np.ndarray, crop_fraction: float = 0.85) -> np.ndarray:
    """
    Effettua un ritaglio centrale dell'immagine mantenendo una percentuale (crop_fraction) della sua dimensione originale.
    Utile per calcolare metriche escludendo i bordi, che spesso contengono rumore.
    """
    h, w = image.shape[:2]
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return image[y0:y0 + ch, x0:x0 + cw]

def blockiness_score(gray: np.ndarray, block_size: int = 8) -> float:
    """
    Calcola un punteggio che indica la presenza di artefatti "a blocchi" (tipici della compressione JPEG)
    confrontando le differenze sui bordi dei blocchi 8x8 con le differenze naturali nell'immagine.
    """
    gray = gray.astype(np.float32)
    vertical = gray[:, block_size::block_size] - gray[:, block_size - 1::block_size]
    horizontal = gray[block_size::block_size, :] - gray[block_size - 1::block_size, :]
    boundary = np.mean(np.abs(vertical)) + np.mean(np.abs(horizontal))
    natural = np.mean(np.abs(np.diff(gray, axis=1))) + np.mean(np.abs(np.diff(gray, axis=0)))
    return float(boundary / (natural + 1e-8))

def quality_metrics(image: np.ndarray) -> dict:
    """
    Estrae le metriche principali sull'immagine (luminosità media/std, percentuale di pixel troppo scuri o troppo chiari,
    varianza del laplaciano per valutare il blur, rumore impulsivo e blockiness).
    """
    crop = center_crop(image)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return {
        'brightness_mean': float(gray.mean()),
        'brightness_std': float(gray.std()),
        'dark_ratio': float((gray < 25).mean()),
        'bright_ratio': float((gray > 230).mean()),
        'laplacian_var': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'impulse_ratio': float(((gray <= 2) | (gray >= 253)).mean()),
        'blockiness': blockiness_score(gray),
    }

def convert_ImageAsset_to_PIL_image(image_asset: ImageAsset) -> Image.Image:
    """
    Converte un oggetto ImageAsset in un'immagine PIL standard in modo
    da poterla utilizzare agevolmente con tutte le funzioni successive.
    """
    # Convert the ImageAsset to bytes
    image_bytes = image_asset.load_bytes() 
    # Create a BytesIO object from the bytes
    image_stream = io.BytesIO(image_bytes)
    # Open the image using PIL
    pil_image = Image.open(image_stream)
    
    return pil_image

def image_quality_assessment(pil_image):
    """
    Valuta la qualità di un'immagine rispetto a diverse famiglie di degradazioni 
    (esposizione, rumore impulsivo, jpeg, blur).
    Restituisce:
    - status: 'good' (ok), 'fixable' (reparabile) o 'bad' (da scartare).
    - reasons: quali tipi di degrado sono stati trovati.
    - metrics: i valori delle metriche.
    """
    # Converte l'immagine PIL in array numpy (RGB)
    image_np = np.array(pil_image.convert('RGB'))
    
    # Calcola le metriche
    metrics = quality_metrics(image_np)
    diagnostics = {}
    
    # Controlli Exposure
    c_exp = TRIGGER_CONFIG['exposure']
    if metrics['brightness_mean'] <= c_exp['dark_bad_mean'] or metrics['dark_ratio'] >= c_exp['dark_bad_ratio'] or \
       metrics['brightness_mean'] >= c_exp['bright_bad_mean'] or metrics['bright_ratio'] >= c_exp['bright_bad_ratio']:
        diagnostics['exposure'] = 'bad'
    elif metrics['brightness_mean'] <= c_exp['dark_fixable_mean'] or metrics['dark_ratio'] >= c_exp['dark_fixable_ratio'] or \
         metrics['brightness_mean'] >= c_exp['bright_fixable_mean'] or metrics['bright_ratio'] >= c_exp['bright_fixable_ratio']:
        diagnostics['exposure'] = 'fixable'
    else:
        diagnostics['exposure'] = 'good'

    # Controlli Impulse Noise
    c_noise = TRIGGER_CONFIG['impulse_noise']
    if metrics['impulse_ratio'] >= c_noise['bad_impulse_ratio']:
        diagnostics['impulse_noise'] = 'bad'
    elif metrics['impulse_ratio'] >= c_noise['fixable_impulse_ratio']:
        diagnostics['impulse_noise'] = 'fixable'
    else:
        diagnostics['impulse_noise'] = 'good'

    # Controlli JPEG Blockiness
    c_jpeg = TRIGGER_CONFIG['jpeg']
    if metrics['blockiness'] >= c_jpeg['bad_blockiness']:
        diagnostics['jpeg'] = 'bad'
    elif metrics['blockiness'] >= c_jpeg['fixable_blockiness']:
        diagnostics['jpeg'] = 'fixable'
    else:
        diagnostics['jpeg'] = 'good'

    # Controlli Blur
    c_blur = TRIGGER_CONFIG['blur']
    if metrics['laplacian_var'] <= c_blur['bad_laplacian_max']:
        diagnostics['blur'] = 'bad'
    elif metrics['laplacian_var'] <= c_blur['fixable_laplacian_max']:
        diagnostics['blur'] = 'fixable'
    else:
        diagnostics['blur'] = 'good'

    # Valutazione globale
    if any(status == 'bad' for status in diagnostics.values()):
        overall_status = 'bad'
    elif any(status == 'fixable' for status in diagnostics.values()):
        overall_status = 'fixable'
    else:
        overall_status = 'good'
        
    # Filtra solo i problemi per indicare che fix applicare
    reasons = {k: v for k, v in diagnostics.items() if v != 'good'}
    
    return {
        'status': overall_status,
        'reasons': reasons,
        'metrics': metrics
    }

def lab_luminance_to_target(image: np.ndarray, target_mean: float, max_up: float = 1.35, max_down: float = 0.75) -> np.ndarray:
    """
    Corregge l'esposizione dell'immagine agendo sul canale della luminosità in spazio LAB.
    Regola l'immagine avvicinandola a una luminosità target, rispettando un limite massimo all'innalzamento/abbassamento.
    """
    current = quality_metrics(image)['brightness_mean']
    scale = np.clip(target_mean / (current + 1e-8), max_down, max_up)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] * scale, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applica un filtro mediano. 
    Efficace per rimuovere rumore impulsivo ("salt and pepper") senza distruggere i bordi visivi.
    """
    return cv2.medianBlur(image, int(ksize))

def jpeg_deblock_bilateral(image: np.ndarray, d: int = 3, sigma_color: float = 20, sigma_space: float = 3) -> np.ndarray:
    """
    Applica un filtro bilaterale per ridurre la "blockiness" del formato JPEG.
    Ammorbidisce le zone sfocate ma preserva i contorni forti dell'immagine.
    """
    return cv2.bilateralFilter(image, int(d), float(sigma_color), float(sigma_space))

def unsharp_mild(image: np.ndarray, amount: float = 0.20, sigma: float = 1.0) -> np.ndarray:
    """
    Applica un filtro di "Unsharp Masking" leggero.
    Aiuta a recuperare un minimo di nitidezza nei casi di lieve sfocatura (blur focale/gaussiano).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return cv2.addWeighted(image, 1 + float(amount), blurred, -float(amount), 0)

DEFAULT_PREPROCESSING_PARAMS = {
    'exposure': {'max_up': 1.35, 'max_down': 0.75},
    'impulse_noise': {'ksize': 3},
    'jpeg': {'d': 3, 'sigma_color': 15, 'sigma_space': 3},
    'blur': {'amount': 0.25, 'sigma': 1.0},
}

def fix_image(pil_image: Image.Image, reasons: dict, brightness_target: float = 100.0, params: dict = None) -> Image.Image:
    """
    Applica i fix necessari a un'immagine basandosi sul dizionario 'reasons'
    restituito dalla funzione image_quality_assessment.
    Restituisce la nuova immagine in formato PIL Image.
    """
    image_np = np.array(pil_image.convert('RGB'))
    fixed_image_np = image_np.copy()
    
    if params is None:
        params = DEFAULT_PREPROCESSING_PARAMS
        
    for family, status in reasons.items():
        if status == 'fixable':
            fam_params = params.get(family, DEFAULT_PREPROCESSING_PARAMS[family])
            if family == 'exposure':
                fixed_image_np = lab_luminance_to_target(
                    fixed_image_np, brightness_target, 
                    max_up=fam_params['max_up'], max_down=fam_params['max_down']
                )
            elif family == 'impulse_noise':
                fixed_image_np = median_filter(fixed_image_np, ksize=fam_params['ksize'])
            elif family == 'jpeg':
                fixed_image_np = jpeg_deblock_bilateral(
                    fixed_image_np, d=fam_params['d'], 
                    sigma_color=fam_params.get('sigma_color', 20), 
                    sigma_space=fam_params.get('sigma_space', 3)
                )
            elif family == 'blur':
                fixed_image_np = unsharp_mild(
                    fixed_image_np, amount=fam_params['amount'], sigma=fam_params['sigma']
                )
                
    return Image.fromarray(fixed_image_np)

def preprocess_image(image_asset: ImageAsset) -> tuple[Image.Image, str]:
    """
    Funzione unica di livello superiore progettata per essere utilizzata
    come tool principale da un agente o in automatico.

    Riceve in input un 'image_asset' (un file o un object lettore compatibile).
    Esegue l'intera pipeline di valutazione e potenziale fissaggio.
    
    Ritorna una tupla:
    1. L'immagine in formato PIL.Image (originale se good/bad, corretta se fixable).
    2. Identificatore di stato: "USABLE" se è valida per la CNN, "DISCARD" se è da scartare.
    """
    # 1. Converte in PIL 
    pil_image = convert_ImageAsset_to_PIL_image(image_asset)
    
    # 2. Valuta i difetti
    assessment = image_quality_assessment(pil_image)
    status = assessment['status']
    
    # 3. Restituisce il risultato secondo lo stato trovato
    if status == 'bad':
        return pil_image, 'DISCARD'
    elif status == 'fixable':
        fixed_pil_image = fix_image(pil_image, assessment['reasons'])
        return fixed_pil_image, 'USABLE'
    else:  # status == 'good'
        return pil_image, 'USABLE'


    