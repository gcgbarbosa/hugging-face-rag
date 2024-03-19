import pandas as pd


class DrugDatabaseInfo:
    df = pd.read_csv("data/drugsES_traindataset_20240227.csv")

    def find_information(self, drug_name: str):
        """
        Finds information about a drug by its name in the database.
        
        Parameters:
            drug_name (str): The name of the drug to search for.
        
        Returns:
            A pandas Series with the information found or None if no records were found.
        """
        drug_name = drug_name.lower()

        # Search by drug name
        records_by_drug = self.df.nombre_del_medicamento.str.lower().str.contains(
            drug_name
        )
        # Search by active substance name
        records_by_active_substance = (
            self.df.sustancias_activas.str.lower().str.contains(drug_name)
        )

        # Find records that match either the drug name or the active substance name
        records = self.df[records_by_active_substance | records_by_drug]

        if len(records) > 0:
            """
            If there are records, return the first one (there should never be more than one record
            with the same drug or active substance name, but we're not taking any chances).
            """
            return records.iloc[0]

        return None
