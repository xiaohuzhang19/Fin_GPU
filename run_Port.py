import os
import pandas as pd
import time
from src.models.mc import hybridMonteCarlo
from src.models.longstaff import LSMC_Numpy, LSMC_OpenCL
import src.models.benchmarks as bm
from src.models.pso import PSO_Numpy, PSO_OpenCL
from src.models.utils import checkOpenCL
import argparse

#===config file=====

nPath = 10000
nPeriod = 100
nFish = 5000

def run_models_on_row(row):
    T = row['days']/ 365
    S0 = row['forward_price']
    r = row['_3_MO'] / 100  # Convert % to decimal
    sigma = row['impl_volatility']
    K = row['strike_price']
    opttype = row['cp_flag'].upper()

    timings = {}
    
    try:
        # Monte Carlo Initialization
        start_mc = time.time()
        mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
        timings['mc_setup'] = time.time() - start_mc

        # Binomial
        start_binomial = time.time()
        binomial = bm.binomialAmericanOption(S0, K, r, sigma, nPeriod, T, opttype)
        timings['binomial'] = time.time() - start_binomial

        # LSMC Numpy
        start_lsmc_np = time.time()
        lsmc_np = LSMC_Numpy(mc)
        lsmc_val_np = float(lsmc_np.longstaff_schwartz_itm_path_fast()[0])
        timings['lsmc_np'] = time.time() - start_lsmc_np

        # LSMC OpenCL
        start_lsmc_cl = time.time()
        lsmc_cl = LSMC_OpenCL(mc)
        lsmc_val_cl = float(lsmc_cl.longstaff_schwartz_itm_path_fast_hybrid()[0])
        timings['lsmc_cl'] = time.time() - start_lsmc_cl

        # PSO Numpy
        start_pso_np = time.time()
        pso_np = PSO_Numpy(mc, nFish, mc.costPsoAmerOption_np)
        pso_val_np = pso_np.solvePsoAmerOption_np()
        timings['pso_np'] = time.time() - start_pso_np

        # PSO OpenCL
        start_pso_cl = time.time()
        pso_cl = PSO_OpenCL(mc, nFish, mc.costPsoAmerOption_cl)
        pso_val_cl = pso_cl.solvePsoAmerOption_cl()
        timings['pso_cl'] = time.time() - start_pso_cl

        # Cleanup
        pso_cl.cleanUp()
        mc.cleanUp()

        return pd.Series([
            binomial, lsmc_val_np, lsmc_val_cl, pso_val_np, pso_val_cl,
            timings['mc_setup'], timings['binomial'], timings['lsmc_np'],
            timings['lsmc_cl'], timings['pso_np'], timings['pso_cl']
        ])

    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series([None]*11)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run pricing models on a CSV input file")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("output_file", type=str, help="Path to output CSV file")
    args = parser.parse_args()
    # Construct full paths
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the base path to the 'Data' directory
    base_path = os.path.join(script_dir, 'Data')

    # Construct full paths
    input_path = args.input_file if os.path.isabs(args.input_file) else os.path.join(base_path, args.input_file)
    output_path = args.output_file if os.path.isabs(args.output_file) else os.path.join(base_path, args.output_file)
    checkOpenCL()
    
    columns_to_keep = [
    'secid', 'date', 'cp_flag', 'days', 
    'impl_volatility', 'strike_price', 'premium',
     '_3_MO', 'forward_price'
    ]   
    # Load and filter input data
    df = pd.read_csv(input_path)
    df = df[columns_to_keep]

    # Apply models row-by-row
    print("Running pricing models on each row...")

    model_columns = ['binomial', 'lsmc_cpu', 'lsmc_gpu', 'pso_cpu', 'pso_gpu',
                     'time_mc_setup', 'time_binomial', 'time_lsmc_cpu',
                     'time_lsmc_gpu', 'time_pso_cpu', 'time_pso_gpu']

    # Apply and rename
    model_results = df.apply(run_models_on_row, axis=1)
    model_results.columns = model_columns

    # Combine original data + model output
    df_final = pd.concat([df, model_results], axis=1)

    # Save to output
    if output_path.endswith('.xlsx'):
        import openpyxl
        df_final.to_excel(output_path, index=False, engine='openpyxl')
        print(f"✅ Done. Results saved to Excel file: {output_path}")
    else:
        df_final.to_csv(output_path, index=False)
        print(f"✅ Done. Results saved to CSV file: {output_path}")
