from ete3 import NCBITaxa

ncbi = NCBITaxa()

def get_taxonomic_lineage_info(taxid):
    """
    Given an NCBI TaxID, return two dictionaries:
      1) rank_to_taxid: { rank -> taxid }
      2) rank_to_name: { rank -> taxonomic name }
    """

    # Step 1: get lineage TaxIDs
    lineage_taxids = ncbi.get_lineage(taxid)

    # Step 2: get rank information
    taxid_to_rank = ncbi.get_rank(lineage_taxids)   # {taxid: rank}

    # Step 3: get names for all lineage taxids
    taxid_to_name = ncbi.get_taxid_translator(lineage_taxids)  # {taxid: name}

    # Step 4: build rank→taxid and rank→name mappings
    rank_to_taxid = {}
    rank_to_name = {}

    for tid in lineage_taxids:
        rank = taxid_to_rank.get(tid, None)
        if rank and rank != "no rank":
            rank_to_taxid[rank] = tid
            rank_to_name[rank] = taxid_to_name[tid]

    return rank_to_taxid, rank_to_name

if __name__ == "__main__":
  print(get_taxonomic_lineage_info(1823908))
  print(get_taxonomic_lineage_info(1777767))

