import os
import re
import fitz  
import pytesseract
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import logging
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import gc
import time
from pyzbar.pyzbar import decode


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CODE_PATTERN = r'\b\d{15}\b'

INPUT_FOLDER = r"C:\Users\loren\Desktop\Script\pdf_originali"
OUTPUT_FOLDER = r"C:\Users\loren\Desktop\Script\pdf_rinominati"
LOG_FILE = r"C:\Users\loren\Desktop\Script\processing.log"
CHECKPOINT_FILE = r"C:\Users\loren\Desktop\Script\checkpoint.json"
DEBUG_FOLDER = r"C:\Users\loren\Desktop\Script\debug_images"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def is_valid_code(code):
    if not code or not isinstance(code, str):
        return False
    
    if not code.isdigit() or len(code) != 15:
        return False
    
    if code == '0' * 15:
        return False
    
    if len(set(code)) < 3:
        return False
    
    return True


def score_code_quality(code):
    score = 0
    
    if code.startswith('4'):
        score += 100
    if code.startswith(('45', '46', '47', '48', '49')):
        score += 50
    
    unique_digits = len(set(code))
    score += unique_digits * 10
    
    if unique_digits < 5:
        score -= 50
    
    for digit in '0123456789':
        if digit * 4 in code:
            score -= 30
    
    return score


def find_barcode_region(img, debug=False):
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    crop_top = int(height * 0.60)
    
    if debug:
        print(f"\n Dimensioni originali: {width}x{height}")
        print(f" Prendo solo dal pixel {crop_top} in giù (ultimi 40%)")
    
    cropped = img.crop((0, crop_top, width, height))
    
    if debug:
        print(f" Nuove dimensioni: {cropped.size}")
    
    return cropped


def find_barcode_region_smart(img, debug=False):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    default_crop = int(height * 0.60)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    barcode_candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if y < height * 0.5:
            continue
        
        if w < width * 0.3 or w > width * 0.9:
            continue
        if h < 50 or h > 300:
            continue
        
        aspect = w / h
        if aspect < 2 or aspect > 8:
            continue
        
        barcode_candidates.append((x, y, w, h, y))
    
    if barcode_candidates:
        barcode_candidates.sort(key=lambda x: x[4], reverse=True)
        x, y, w, h, _ = barcode_candidates[0]
        
        margin = 20
        y1 = max(0, y - margin)
        y2 = min(height, y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(width, x + w + margin)
        
        if debug:
            print(f"Trovato barcode box: x={x}, y={y}, w={w}, h={h}")
            print(f"Crop da y={y1} a y={y2}")
        
        return img.crop((x1, y1, x2, y2))
    
    if debug:
        print(f"Nessun box trovato, uso crop default (40% inferiore)")
    
    return img.crop((0, default_crop, width, height))


def preprocess_for_barcode(img, debug=False):
    try:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        debug_steps = {}
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        if debug:
            debug_steps['1_gray'] = Image.fromarray(gray)
        
        height, width = gray.shape
        if height < 300:
            scale = 400 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            if debug:
                debug_steps['2_resized'] = Image.fromarray(gray)
        
        try:
            denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        except:
            denoised = gray
        
        if debug:
            debug_steps['3_denoised'] = Image.fromarray(denoised)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        if debug:
            debug_steps['4_enhanced'] = Image.fromarray(enhanced)
        
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if debug:
            debug_steps['5_binary'] = Image.fromarray(binary)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if debug:
            debug_steps['6_morph'] = Image.fromarray(morph)
        
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(morph, -1, kernel_sharp)
        
        if debug:
            debug_steps['7_final'] = Image.fromarray(sharpened)
        
        result = Image.fromarray(sharpened)
        
        if debug:
            return result, debug_steps
        return result, None
        
    except Exception as e:
        if debug:
            print(f"  Errore pre-processing: {e}")
        return img, None


def extract_all_15digit_codes(text, debug=False):
    if not text:
        return []
    
    text_clean = ''.join(c if c.isdigit() or c.isspace() else ' ' for c in text)
    text_clean = ' '.join(text_clean.split())
    
    if debug:
        print(f"  Testo pulito: {text_clean[:200]}")
    
    valid_codes = []
    
    matches = re.findall(CODE_PATTERN, text_clean)
    for match in matches:
        if is_valid_code(match) and match not in valid_codes:
            valid_codes.append(match)
            if debug:
                print(f" Codice (regex): {match}")
    
    text_no_spaces = text_clean.replace(' ', '')
    if len(text_no_spaces) >= 15:
        for i in range(len(text_no_spaces) - 14):
            candidate = text_no_spaces[i:i+15]
            if is_valid_code(candidate) and candidate not in valid_codes:
                valid_codes.append(candidate)
                if debug:
                    print(f"  Codice (sliding): {candidate}")
    
    return valid_codes


def extract_code_from_pdf(pdf_path, debug=False):
    try:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            return None, "PDF vuoto"

        page = doc[0]
        mat = fitz.Matrix(4, 4)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        barcodes = decode(img_cv)
        found_codes = set()
        for barcode in barcodes:
            code = barcode.data.decode("utf-8").strip()
            if is_valid_code(code):
                found_codes.add(code)
                if debug:
                    print(f" Barcode decodificato da pyzbar: {code}")

        if found_codes:
            best = max(found_codes, key=score_code_quality)
            return best, None

        h = img.height
        img_crop = img.crop((0, int(h * 0.6), img.width, h))

        gray = cv2.cvtColor(np.array(img_crop), cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        ocr_text = pytesseract.image_to_string(
            gray,
            lang='eng',
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        )
        if debug:
            print("Testo OCR:", repr(ocr_text))

        codes = extract_all_15digit_codes(ocr_text, debug=debug)
        if not codes:
            return None, "Nessun codice trovato"

        best_code = max(codes, key=score_code_quality)
        return best_code, None

    except Exception as e:
        if debug:
            print(f"Errore durante estrazione: {e}")
        gc.collect()
        return None, str(e)
        
    except Exception as e:
        if doc:
            doc.close()
        logging.exception(f"Errore estrazione da {Path(pdf_path).name}:")
        return None, f"Errore: {str(e)}"


def append_to_mapping_file(output_folder, codice_anagrafica, codice_fiscale):
    mapping_file = Path(output_folder) / "abbinamenti_codici.txt"
    
    try:
        with open(mapping_file, 'a', encoding='utf-8') as f:
            f.write(f"{codice_anagrafica} -> {codice_fiscale}\n")
    except Exception as e:
        logging.error(f"Errore nella scrittura del file mapping: {e}")


def extract_codice_fiscale_from_pdf(pdf_path, codice_anagrafica, debug=False):
    try:
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            return "ERROR"

        page = doc[0]
        mat = fitz.Matrix(4, 4)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        def find_cf_in_barcodes(barcodes, debug=False):
            for barcode in barcodes:
                code = barcode.data.decode('utf-8').strip()
                if debug:
                    print(f"   Barcode: {code}")
                
                has_letters = any(c.isalpha() for c in code)
                
                if has_letters and len(code) == 16:
                    if debug:
                        print(f" CF trovato: {code}")
                    return code
            return None

        if debug:
            print(f" Strategia 1: scansione pagina intera")
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        barcodes = decode(img_cv)
        
        if debug:
            print(f"   Trovati {len(barcodes)} barcode")
        
        cf = find_cf_in_barcodes(barcodes, debug)
        if cf:
            return cf

        if debug:
            print(f" Strategia 2: metà superiore")
        
        h = img.height
        img_top = img.crop((0, 0, img.width, int(h * 0.5)))
        img_cv_top = cv2.cvtColor(np.array(img_top), cv2.COLOR_RGB2BGR)
        barcodes_top = decode(img_cv_top)
        
        if debug:
            print(f"   Trovati {len(barcodes_top)} barcode")
        
        cf = find_cf_in_barcodes(barcodes_top, debug)
        if cf:
            return cf

        if debug:
            print(f" Strategia 3: metà inferiore")
        
        img_bottom = img.crop((0, int(h * 0.5), img.width, h))
        img_cv_bottom = cv2.cvtColor(np.array(img_bottom), cv2.COLOR_RGB2BGR)
        barcodes_bottom = decode(img_cv_bottom)
        
        if debug:
            print(f"   Trovati {len(barcodes_bottom)} barcode")
        
        cf = find_cf_in_barcodes(barcodes_bottom, debug)
        if cf:
            return cf

        if debug:
            print(f" Strategia 4: risoluzione massima")
        
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        mat = fitz.Matrix(6, 6)
        pix = page.get_pixmap(matrix=mat)
        img_hires = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        
        img_cv_hires = cv2.cvtColor(np.array(img_hires), cv2.COLOR_RGB2BGR)
        barcodes_hires = decode(img_cv_hires)
        
        if debug:
            print(f"   Trovati {len(barcodes_hires)} barcode")
        
        cf = find_cf_in_barcodes(barcodes_hires, debug)
        if cf:
            return cf
        
        if debug:
            print(f" Nessun CF trovato dopo tutte le strategie")
        return "ERROR"
            
    except Exception as e:
        if debug:
            print(f"Errore nell'estrazione del codice fiscale: {e}")
        return "ERROR"


def process_single_pdf(pdf_path, output_folder, debug=False):
    try:
        code, error = extract_code_from_pdf(pdf_path, debug=debug)
        
        if error:
            logging.error(f" {Path(pdf_path).name}: {error}")
            return False, error
        
        if not code:
            logging.error(f" {Path(pdf_path).name}: Nessun codice")
            return False, "Codice non trovato"
        
        if not debug:
            logging.info(f" {Path(pdf_path).name}: {code}")
        
        new_name = f"{code}.pdf"
        new_path = Path(output_folder) / new_name
        
        counter = 1
        while new_path.exists():
            new_name = f"{code}_{counter}.pdf"
            new_path = Path(output_folder) / new_name
            counter += 1
        
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            
            if Path(pdf_path).parent == Path(output_folder):
                Path(pdf_path).rename(new_path)
            else:
                shutil.copy2(pdf_path, new_path)
            
            if debug:
                print(f"\n Salvato: {new_name}")

            codice_fiscale = extract_codice_fiscale_from_pdf(new_path, code, debug=debug)
            
            append_to_mapping_file(output_folder, code, codice_fiscale)
            
            if debug:
                print(f" Abbinamento salvato: {code} -> {codice_fiscale}")
            
            return True, code
            
        except Exception as e:
            logging.error(f" Errore salvataggio {Path(pdf_path).name}: {e}")
            return False, str(e)
            
    except Exception as e:
        logging.exception(f" {Path(pdf_path).name}")
        return False, str(e)


def save_checkpoint(processed_files):
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(processed_files), f, indent=2)
    except Exception as e:
        logging.warning(f"Errore checkpoint: {e}")


def load_checkpoint():
    try:
        if Path(CHECKPOINT_FILE).exists():
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.info(f"Checkpoint: {len(data)} file processati")
                return set(data)
    except Exception as e:
        logging.warning(f" Errore load checkpoint: {e}")
    return set()


def process_batch_parallel(input_folder, output_folder, max_workers=None):
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    
    if not pdf_files:
        logging.error(f" Nessun PDF in {input_folder}")
        return
    
    processed_files = load_checkpoint()
    remaining_files = [f for f in pdf_files if f.name not in processed_files]
    
    if not remaining_files:
        logging.info(" Tutti già processati!")
        return
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING - ZONA BARCODE")
    print(f"{'='*80}")
    print(f"Totali:      {len(pdf_files)}")
    print(f"Processati:  {len(processed_files)}")
    print(f"Rimanenti:   {len(remaining_files)}")
    print(f"{'='*80}\n")
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    print(f"Worker: {max_workers}\n")
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf_path, output_folder, False): pdf_path
            for pdf_path in remaining_files
        }
        
        completed = 0
        total = len(futures)
        
        for future in futures:
            try:
                success, result = future.result(timeout=180)
                results.append((success, result, futures[future].name))
                completed += 1
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed > 0 else 0
                remaining_time = avg_time * (total - completed)
                
                status = "" if success else ""
                percent = (completed / total) * 100
                print(f"\r{status} [{completed}/{total}] {percent:.1f}% | "
                      f"Rimanente: {remaining_time/60:.1f} min      ", 
                      end='', flush=True)
                
                if success:
                    processed_files.add(futures[future].name)
                    if completed % 10 == 0:
                        save_checkpoint(processed_files)
                        
            except Exception as e:
                logging.error(f" Errore: {e}")
                results.append((False, str(e), 'unknown'))
                completed += 1
        
        print()
    
    save_checkpoint(processed_files)
    
    successi = sum(1 for r in results if r[0])
    fallimenti = len(results) - successi
    total_time = (time.time() - start_time) / 60
    
    print(f"\n\n{'='*80}")
    print(f"REPORT FINALE")
    print(f"{'='*80}")
    print(f"Processati:  {len(results)}")
    print(f" Successi:  {successi}/{len(results)} ({successi/len(results)*100:.1f}%)")
    print(f" Fallimenti: {fallimenti}/{len(results)} ({fallimenti/len(results)*100:.1f}%)")
    print(f"  Tempo:     {total_time:.1f} min")
    print(f"  Medio:     {total_time*60/len(results):.1f} sec/file")
    print(f"{'='*80}")

    mapping_file = Path(output_folder) / "abbinamenti_codici.txt"
    if mapping_file.exists():
        print(f"\n Abbinamenti salvati in: {mapping_file}")
    errori = [(r[2], r[1]) for r in results if not r[0]]
    
    if errori:
        print(f"\nERRORI ({len(errori)}):")
        for idx, (file, err) in enumerate(errori[:10], 1):
            print(f"  {idx}. {file}: {err}")
        
        if len(errori) > 10:
            print(f"  ... e altri {len(errori)-10}")
        
        error_file = Path(output_folder) / "errori.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            for file, err in errori:
                f.write(f"{file}\t{err}\n")
        print(f"\n Errori: {error_file}")
    else:
        print(f"\n TUTTI SUCCESSO ")


if __name__ == "__main__":
    print("="*80)
    print("OCR PDF - ESTRAZIONE CODICE DA BARCODE")
    print("="*80)
    print("\n1. TEST  - Singolo file debug")
    print("2. BATCH - Tutti i PDF")
    
    try:
        mode = input("\nModalità (1/2): ").strip()
    except:
        mode = "1"
    
    if mode == "1":
        print(f"\n{'='*80}")
        print("MODALITÀ TEST")
        print(f"{'='*80}\n")
        
        all_pdfs = list(Path(INPUT_FOLDER).glob("*.pdf"))
        
        if not all_pdfs:
            print(f" Nessun PDF in {INPUT_FOLDER}")
        else:
            print(f"PDF disponibili ({len(all_pdfs)}):")
            for idx, pdf in enumerate(all_pdfs[:10], 1):
                print(f"  {idx}. {pdf.name}")
            if len(all_pdfs) > 10:
                print(f"  ... e altri {len(all_pdfs)-10}")
            
            print(f"\nINVIO per primo file o scrivi nome:")
            try:
                user_input = input("> ").strip()
            except:
                user_input = ""
            
            test_pdf = None
            if user_input:
                for pdf in all_pdfs:
                    if pdf.name == user_input or pdf.stem == user_input:
                        test_pdf = pdf
                        break
                if not test_pdf:
                    test_pdf = all_pdfs[0]
            else:
                test_pdf = all_pdfs[0]
            
            if test_pdf and test_pdf.exists():
                print(f"\n{'='*80}")
                print(f" ANALISI: {test_pdf.name}")
                print(f"{'='*80}\n")
                
                if Path(DEBUG_FOLDER).exists():
                    try:
                        shutil.rmtree(DEBUG_FOLDER)
                    except:
                        pass
                Path(DEBUG_FOLDER).mkdir(parents=True, exist_ok=True)
                
                success, result = process_single_pdf(test_pdf, OUTPUT_FOLDER, debug=True)
                
                print(f"\n{'='*80}")
                if success:
                    print(f" SUCCESSO ")
                    print(f"{'='*80}")
                    print(f"Codice: {result}")
                    print(f"\n Debug images: {DEBUG_FOLDER}")
                else:
                    print(f" FALLITO ")
                    print(f"{'='*80}")
                    print(f"Errore: {result}")
                    print(f"\n Controlla: {DEBUG_FOLDER}")
                print(f"{'='*80}")
    
    elif mode == "2":
        print(f"\n{'='*80}")
        print("MODALITÀ BATCH")
        print(f"{'='*80}\n")
        
        try:
            confirm = input(f"Processare '{INPUT_FOLDER}'? (y/n): ").strip().lower()
        except:
            confirm = "n"
        
        if confirm == "y":
            process_batch_parallel(INPUT_FOLDER, OUTPUT_FOLDER)
        else:
            print("Annullato")
    
    print(f"\n{'='*80}")
    print("FINE")
    print(f"{'='*80}")
