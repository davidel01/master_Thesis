from collections import Counter, defaultdict
import timeout_decorator
from tqdm import tqdm
import pandas as pd
import hashlib
from rapidfuzz.distance import LCSseq
import bisect
from itertools import product

def compute_bag_intersection(r_tab, s_tab, seed_ids, seeds):
    col_pairs = [seeds[i][0] for i in seed_ids] # O(k)
    col_pairs.sort(key=lambda s: s[0], reverse=False) # O(k log k)

    r_bag = to_bag_counter([r_tab[i] for i in [c_p[0] for c_p in col_pairs]]) # O(n * k)
    s_bag = to_bag_counter([s_tab[i] for i in [c_p[1] for c_p in col_pairs]]) # O(n * k)

    return r_bag & s_bag # O(n)

def longest_common_subsequence(seq1: list, seq2: list):
    m, n = len(seq1), len(seq2)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                f[i][j] = f[i - 1][j - 1] + 1
            else:
                f[i][j] = max(f[i - 1][j], f[i][j - 1])
    
    indices_seq1 = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            indices_seq1.append(i - 1)
            i -= 1
            j -= 1
        elif f[i - 1][j] > f[i][j - 1]:
            i -= 1
        else:
            j -= 1
    indices_seq1.reverse()

    # Return length and index of the subsequence
    return f[m][n], indices_seq1

def compute_long_comm_subseq(r_tab, s_tab, seed_ids, seeds, index=False):
    col_pairs = [seeds[i][0] for i in seed_ids] # O(k)
    col_pairs.sort(key=lambda s: s[0], reverse=False) # O(k log k)

    new_r_tab = [r_tab[m[0]] for m in col_pairs] # O(k)
    new_s_tab = [s_tab[m[1]] for m in col_pairs] # O(k)

    # Combine corresponded elements (i.e. in the same row) for specific columns
    combined_r = ["".join(map(str,items)) for items in zip(*new_r_tab)] # O(n * k)
    combined_s = ["".join(map(str,items)) for items in zip(*new_s_tab)] # O(n * k)

    # Compute the LCS algorithm on the two sequences made
    if index:
        length, idx = longest_common_subsequence(combined_r, combined_s) # O(n^2)
        return length, idx
    
    length = LCSseq.similarity(combined_r, combined_s) # O(n^2)
    return length

# Versione più pulita e ottimizzata
def compute_long_comm_subseq_best(r_tab, s_tab, seed_ids, seeds):
    """
    Versione ottimizzata finale: combina le migliori ottimizzazioni
    """
    # Estrai indici una sola volta
    col_pairs = [seeds[i][0] for i in seed_ids]
    # Costruisci sequenze usando list comprehension più efficiente
    n_rows = len(r_tab[0]) if r_tab else 0
    combined_r = [
        tuple(r_tab[col_pair[0]][row] for col_pair in col_pairs)
        for row in range(n_rows)
    ]
    combined_s = [
        tuple(s_tab[col_pair[1]][row] for col_pair in col_pairs)
        for row in range(n_rows)
    ]
    return LCSseq.similarity(combined_r, combined_s)

def longest_common_substring(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    max_len = 0
    end_seq1 = 0
    end_seq2 = 0

    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > max_len:
                    max_len = dp[i + 1][j + 1]
                    end_seq1 = i + 1
                    end_seq2 = j + 1
            else:
                dp[i + 1][j + 1] = 0

    start_seq1 = end_seq1 - max_len
    start_seq2 = end_seq2 - max_len
    return max_len, start_seq1

def longest_common_substr_space_opt(s1, s2):
    m, n = len(s1) , len(s2)
    result = 0

    # Matrix to store result of two
    # consecutive rows at a time.
    dp = [[0] * (n + 1) for _ in range(2)]

    # Variable to represent which row of
    # matrix is current row.
    currRow = 0

    # For a particular value of i and j,
    # dp[currRow][j] stores length of longest
    # common substring in string X[0..i] and Y[0..j].
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[currRow][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                dp[currRow][j] = dp[1 - currRow][j - 1] + 1
                result = max(result, dp[currRow][j])
            else:
                dp[currRow][j] = 0
        # Make current row as previous row and previous
        # row as new current row.
        currRow = 1 - currRow
    return result

def compute_long_comm_substr(r_tab, s_tab, seed_ids, seeds, index=False):
    col_pairs = [seeds[i][0] for i in seed_ids] # O(k)
    col_pairs.sort(key=lambda s: s[0], reverse=False) # O(k log k)

    new_r_tab = [r_tab[m[0]] for m in col_pairs] # O(k)
    new_s_tab = [s_tab[m[1]] for m in col_pairs] # O(k)

    # Combine corresponded elements (i.e. in the same row) for specific columns
    combined_r = ["".join(map(str,items)) for items in zip(*new_r_tab)] # O(n * k)
    combined_s = ["".join(map(str,items)) for items in zip(*new_s_tab)] # O(n * k)

    if index:
        length, idx = longest_common_substring(combined_r, combined_s) # O(n*m)
        return length, idx

    length = longest_common_substr_space_opt(combined_r, combined_s)
    return length

# Restructure the table as the list of its columns, ignoring the headers
def parse_table(table, num_cols, num_headers):
    return [[row[i] for row in table[num_headers:]] for i in range(0, num_cols)]

# Convert a table into a bag of tuples
def to_bag(table):
    counter = dict()
    tuples = [tuple([col[i] for col in table]) for i in range(0, len(table[0]))]
    for i in range(0, len(tuples)):
        if tuples[i] in counter:
            counter[tuples[i]] += 1
        else:
            counter[tuples[i]] = 0
        tuples[i] += (counter[tuples[i]],)
    return set(tuples)

# Convert a table into a bag of tuples using counter objects
def to_bag_counter(table):
    counter = Counter()
    for t in [tuple([col[i] for col in table]) for i in range(0, len(table[0]))]:
        counter[t] += 1
    return counter

def parse_table_from_dataframe(table):
    return [table[col].astype(str).tolist() for col in table.columns]

def generate_table_dict(collection_directory: str, tab_list: list) -> dict:
    metadata_path = collection_directory+'/'+'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata['_id'].isin(tab_list)]
    csv_path = collection_directory+'/'+'tables/'
    out_dict = {}
    for r in tqdm(range(metadata.shape[0])):
        meta = metadata.iloc[r]
        t_name = meta.loc['_id']
        n_header = meta.loc['num_header_rows']
        out_dict[t_name] = pd.read_csv(csv_path+'/'+t_name, dtype=str, header=None, skiprows=n_header)
    return out_dict

def get_constraints(row_constraint='none', col_constraint='none'):
    """
    Ritorna i vincoli per righe e colonne dati:
    - le stringhe row_constraint e col_constraint.
    I vincoli possibili sono: "none", "ordered", "contiguous".
    """
    valid_values = ["none", "ordered", "contiguous"]
    if row_constraint not in valid_values or col_constraint not in valid_values:
        raise ValueError(f"Valori validi: {valid_values}")
    
    return row_constraint, col_constraint

def input_from_console():
    row = input("Inserisci il vincolo per le righe ('none', 'ordered', 'contiguous'): ").strip().lower()
    col = input("Inserisci il vincolo per le colonne ('none', 'ordered', 'contiguous'): ").strip().lower()
    row_constraint, col_constraint = get_constraints(row_constraint=row, col_constraint=col)

    print(f"Configurazione selezionata: righe = '{row_constraint}', colonne = '{col_constraint}'")
    return row_constraint, col_constraint

def lis_bisect(seq):
    if not seq:
        return []

    n = len(seq)
    tail = []               # Lista dei valori minimi per ciascuna lunghezza
    tail_idx = []           # Indici in seq dei valori in tail
    prev_idx = [-1] * n     # Tracciamento per ricostruzione

    for i, num in enumerate(seq):
        pos = bisect.bisect_left(tail, num)

        if pos == len(tail):
            tail.append(num)
            tail_idx.append(i)
        else:
            tail[pos] = num
            tail_idx[pos] = i

        if pos > 0:
            prev_idx[i] = tail_idx[pos - 1]

    # Ricostruzione della LIS
    lis = []
    k = tail_idx[-1]
    while k != -1:
        lis.append(seq[k])
        k = prev_idx[k]

    return lis[::-1]


def find_matches_between_columns(col1, col2):
    idx_map1 = defaultdict(list)
    idx_map2 = defaultdict(list)

    for i, val in enumerate(col1):
        idx_map1[val].append(i)
    for j, val in enumerate(col2):
        idx_map2[val].append(j)

    common_values = idx_map1.keys() & idx_map2.keys()

    result = []
    for val in common_values:
        for i, j in product(idx_map1[val], idx_map2[val]):
            result.append((i, j))

    return sorted(result)


def fast_check(lista1, lista2):
    """
    Verifica se due liste condividono almeno un elemento.
    Ottimizzato per liste molto lunghe.
    
    Strategia:
    1. Converte la lista più piccola in set (O(min(n,m)))
    2. Itera sulla lista più grande fermandosi al primo match
    
    Complessità:
    - Tempo: O(min(n,m)) nel caso migliore, O(n+m) nel peggiore
    - Spazio: O(min(n,m))
    """
    # Assicurati che lista1 sia la più piccola
    if len(lista1) > len(lista2):
        lista1, lista2 = lista2, lista1
    
    # Converti solo la lista più piccola in set
    set_piccolo = set(lista1)
    
    # Itera sulla lista più grande e fermati al primo match
    for elemento in lista2:
        if elemento in set_piccolo:
            return True
    
    return False

def longest_consecutive_pairs(pairs):
    """
    Trova la sottosequenza più lunga di coppie (f, s)
    tale che ogni elemento successivo sia (f+1, s+1),
    anche se non contiguo nella lista.
    Args:
        pairs (list of tuple): lista di coppie (first, second)

    Returns:
        int: lunghezza della sottosequenza più lunga con incrementi consecutivi
    """
    pair_set = set(pairs)  # Per accesso rapido alle coppie presenti
    dp = {}                # Dizionario per memorizzare la lunghezza massima per ogni coppia

    max_len = 0
    for f, s in pairs:
        prev = (f - 1, s - 1)
        if prev in pair_set:
            dp[(f, s)] = dp[prev] + 1
        else:
            dp[(f, s)] = 1
        max_len = max(max_len, dp[(f, s)])

    return max_len


def longest_increasing_pairs(pairs):
    """
    Trova la lunghezza della sottosequenza più lunga di coppie (f, s)
    in cui sia f che s sono strettamente crescenti.
    Args:
        pairs (list of tuple): lista di coppie (first, second)

    Returns:
        int: lunghezza della sottosequenza più lunga con valori crescenti
    """
    if not pairs:
        return 0

    # Ordina per first crescente, e per second decrescente in caso di pareggio
    pairs.sort(key=lambda x: (x[0], -x[1]))

    # Applica LIS sui secondi valori
    seq = []
    for _, s in pairs:
        idx = bisect.bisect_left(seq, s)
        if idx == len(seq):
            seq.append(s)
        else:
            seq[idx] = s
    return len(seq)


def analisi_duplicati(df1, df2):
    ratios1 = []
    ratios2 = []
    for col in df1.columns:
        n_valori = len(df1[col])
        n_unici = df1[col].nunique(dropna=False)  # considera anche i NaN
        n_duplicati = n_valori - n_unici
        ratio = n_duplicati / n_valori
        ratios1.append(ratio)

    for col in df2.columns:
        n_valori = len(df2[col])
        n_unici = df2[col].nunique(dropna=False)  # considera anche i NaN
        n_duplicati = n_valori - n_unici
        ratio = n_duplicati / n_valori
        ratios2.append(ratio)
    
    # misura aggregata: media delle percentuali di duplicati
    misura_totale = sum(ratios1) / len(ratios1)
    misura_totale += sum(ratios2) / len(ratios2)
    misura_totale /= 2

    return misura_totale


def estimate_mapping_density(df1, df2, cols1=None, cols2=None,
                            threshold_dup=0.3):
    """
    Perform a global overlap analysis between two pandas DataFrames
    to estimate mapping density and recommend the best approach
    (mapping-based vs LCS/LCStr-based).

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        The two tables to compare.
    cols1, cols2 : list of str, optional
        Columns to include in the analysis. If None, all columns are used.
    threshold_low, threshold_high : float
        Decision thresholds for R (shared duplication ratio).

    Returns
    -------
    dict
        Summary statistics and final recommendation for the table pair.
    """
    # If no specific columns are provided, use all
    if cols1 is None:
        cols1 = df1.columns
    if cols2 is None:
        cols2 = df2.columns

    # Flatten selected columns into single lists of values
    T1 = df1[cols1].astype(str).values.flatten()
    T2 = df2[cols2].astype(str).values.flatten()

    # Compute frequency distributions
    f1 = Counter(T1)
    f2 = Counter(T2)

    # Shared values only
    shared = set(f1.keys()) & set(f2.keys())

    # Estimate mapping size
    S_est = sum(f1[v] * f2[v] for v in shared)

    # Normalize
    total_pairs = len(T1) * len(T2)
    R = S_est / total_pairs if total_pairs > 0 else 0

    # Decision rule for the whole table
    if R < threshold_dup:
        recommendation = "Mapping-based approach (low shared duplication)"
    else:
        recommendation = "LCS/LCStr-based approach (high shared duplication)"

    # Summary statistics
    result = {
        "Table1_total_values": len(T1),
        "Table2_total_values": len(T2),
        "Distinct_values_T1": len(f1),
        "Distinct_values_T2": len(f2),
        "Shared_values": len(shared),
        "Estimated_mapping_size": S_est,
        "Mapping_density_ratio": R,
        # "Recommendation": recommendation
    }

    return result
