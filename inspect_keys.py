from src.helper.db_session import get_engine, get_session_factory
from src.helper.config import load_app_config
from src.helper.db_schema import PrebidDoc, PrebidVendorExtraction
from sqlalchemy import select

def inspect():
    config = load_app_config()
    engine = get_engine(config)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        # Check PrebidDoc keys
        doc = session.execute(select(PrebidDoc).where(PrebidDoc.extracted_metadata.is_not(None)).limit(1)).scalar_one_or_none()
        if doc:
            print("PrebidDoc keys:", doc.extracted_metadata.keys())
            print("Sample PrebidDoc metadata:", doc.extracted_metadata)
        else:
            print("No PrebidDoc found")

        # Check PrebidVendorExtraction struct (just to see if we have data)
        valid = session.execute(select(PrebidVendorExtraction).limit(1)).scalar_one_or_none()
        if valid:
            print("PrebidVendorExtraction found:", valid.vendor_name)
        else:
            print("No PrebidVendorExtraction found")

if __name__ == "__main__":
    inspect()
