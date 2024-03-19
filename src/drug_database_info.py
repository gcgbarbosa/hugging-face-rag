import pandas as pd


class DrugDatabaseInfo:
    df = pd.read_csv('data/drugsES_traindataset_20240227.csv')

    def find_information(self, drug_name: str):
        drug_name = drug_name.lower()

        records_by_drug = self.df.nombre_del_medicamento.str.lower().str.contains(drug_name)
        records_by_active_substance = self.df.sustancias_activas.str.lower().str.contains(drug_name)

        records = self.df[records_by_active_substance | records_by_drug]

        if len(records) > 0:
            return records.iloc[0]

        return None