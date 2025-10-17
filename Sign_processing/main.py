from Sign_processing.Map_extractor import MapExtractor, ORB_maps
from pathlib import Path
from Sign_processing.Sign_extractor import Sign_extractor_class
from Sign_processing.cnn import CNNPredictor

if __name__ == "__main__":

    output_dir_maps=Path( "Outputs/Output_from_Map_extractor")
    output_dir_sign= Path("Outputs/Output_from_Sign_extractor")
    APV_plan_GDPR_trygg= "APV_plan_GDPR_trygg/AP22222-gdpr fixed AF Gruppen.pdf"

    Orb = ORB_maps (nfeatures=20000,  ratio=0.75,min_good=12000)
    Maps= MapExtractor (orb_matcher = Orb,  output_dir= output_dir_maps)
    Maps.pdf_To_image (APV_plan_GDPR_trygg)
    Files_sorted =sorted( output_dir_maps.glob("page_*.jpg"),
                         key= lambda p: int(p.stem.split("_") [1]) )
    for image_path in Files_sorted:
        sign_extractor = Sign_extractor_class(image_path=image_path, output_dir=output_dir_sign)
        sign_extractor.extract_signs()
    
    model_path = Path("Sign_processing/demo/cnn.pth")
    classes_path = Path("Sign_processing/demo/classes.json")
    predictor = CNNPredictor(model_path, classes_path)
    predictor.predict("Outputs/Output_from_Sign_extractor/page_7_016.png")
    