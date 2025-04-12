from savReaderWriter import SavReader

with SavReader("../models/xgb_regressor_default_42.sav") as reader:
    records = reader.all()
    print(records)
