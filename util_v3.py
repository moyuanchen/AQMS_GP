import pandas as pd
import numpy as np
def load_aqms_data(filepath):
    """
    Load and process AQMS Excel file containing asset price data.
    
    Args:
        filepath (str): Path to AQMS Excel file
        
    Returns:
        pd.DataFrame: Processed DataFrame with datetime index
    """
    xls = pd.ExcelFile(filepath)
    merged_df = None
    
    for sheet_name in xls.sheet_names:
        # Define sheet-specific parameters
        if sheet_name == "Equity":
            skiprows = [0, 1, 3, 4]
        else:
            skiprows = [0, 1, 2, 4, 5]
            
        # Read sheet with custom parameters
        df = pd.read_excel(
            xls,
            sheet_name=sheet_name,
            skiprows=skiprows,
            header=0
        ).copy()
        
        # Process date column
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        
        # Prefix columns with sheet name to avoid collisions
        df.columns = [f"{sheet_name}_{col}" if col != date_col else col 
                     for col in df.columns]
        
        # Rename date column and merge
        df = df.rename(columns={date_col: 'Date'})
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')
    
    # Final processing
    aqms_df = merged_df.sort_values('Date').reset_index(drop=True)
    aqms_df['Date'] = pd.to_datetime(aqms_df['Date'])
    aqms_df.set_index('Date', inplace=True)
    
    return aqms_df


def load_business_cycle_data(filepath):
    """
    Load and process Business Cycle Excel file containing GDP and CPI data.
    
    Args:
        filepath (str): Path to Business Cycle Excel file
        
    Returns:
        pd.DataFrame: Processed DataFrame with year-end datetime index
    """
    sheet_names = ['GDP', 'CPI']
    merged_df = None
    
    for sheet in sheet_names:
        # Read sheet and drop rows with missing country names
        df = pd.read_excel(filepath, sheet_name=sheet, skiprows=[1])
        df = df[df.iloc[:, 0].notna()].reset_index(drop=True)
        
        # Rename first column to 'Country'
        df = df.rename(columns={df.columns[0]: 'Country'})
        
        # Convert to long format
        long_df = df.melt(
            id_vars='Country', 
            var_name='Year', 
            value_name='Value'
        )
        long_df['Year'] = long_df['Year'].astype(str)
        
        # Create feature names and pivot to wide format
        long_df['Feature'] = f"{sheet}_" + long_df['Country']
        pivot_df = long_df.pivot(
            index='Year', 
            columns='Feature', 
            values='Value'
        )
        
        # Merge sheets
        if merged_df is None:
            merged_df = pivot_df
        else:
            merged_df = merged_df.join(pivot_df, how='outer')
    
    # Set year-end datetime index
    bc_df = merged_df.sort_index()
    bc_df.index = pd.to_datetime(bc_df.index.astype(str)) + pd.offsets.YearEnd(0)
    
    return bc_df


def preprocess_asset_data(aqms_df):
    """
    Preprocess asset data to calculate annual returns and yields.
    
    Args:
        aqms_df (pd.DataFrame): Raw AQMS DataFrame
        
    Returns:
        pd.DataFrame: Processed annual returns and yields
    """
    # Define asset categories
    equity = ['US', 'UK', 'Japan', 'Hong Kong', 'Canada', 'Euro', 'Switzerland', 'New Zealand', 'Australia']
    
    int_future = ['SFRA Comdty', 'SFR1YZ2 Comdty', 'SFR1YZ3 Comdty', 
                  'SFR1YZ4 Comdty', 'SFR1YZ5 Comdty', 'SFR1YZ6 Comdty', 
                  'SFR2YZ2 Comdty']
    
    raw = ['JP', 'EU', 'HK', 'CH', 'CA', 'AU', 'NZ']
    currency = raw + ['GBP']
    gov_bond = raw + ['US', 'UK']
    
    # Initialize lists for asset columns
    gb2y, gb10y, curr, eqt, irf = [], [], [], [], []
    
    # Categorize columns
    for name in aqms_df.columns:
        for e in equity:
            if (e in name) and ('Equity' in name) and ('Pan' not in name):
                eqt.append(name)
                
        for g in gov_bond:
            if (g in name) and ('2' in name) and ('Bond' in name):
                gb2y.append(name)
            if (g in name) and ('10' in name) and ('Bond' in name):
                gb10y.append(name)
                
        for c in currency:
            if (c in name) and ('Curncy' in name):
                curr.append(name)
                
        for i in int_future:
            if (i in name) and ('Future' in name):
                irf.append(name)
    
    # Process interest rate columns
    rate_mapping = {
        'US': ['FDTR Index'],
        'UK': ['UKBRBASE Index'],
        'JP': ['BOJDTR Index'],
        'EU': ['EURR002W Index'],
        'HK': ['PRIMHK Index'],
        'CA': ['CABROVER Index'],
        'AU': ['RBATCTR Index'],
        'NZ': ['NZOCR Index'],
        'CH': ['SZNBPOLR Index']
    }
    
    # Create renaming dictionary
    renaming_dict = {}
    for col in aqms_df.columns:
        for abbr, indices in rate_mapping.items():
            for index in indices:
                if index in col:
                    renaming_dict[col] = col.replace(index, abbr)
    
    # Rename columns and get interest rate columns
    aqms_renamed = aqms_df.rename(columns=renaming_dict)
    ir = aqms_renamed[list(renaming_dict.values())]
    
    # Combine all asset data
    assets = pd.concat([
        aqms_renamed[eqt], 
        aqms_renamed[gb2y], 
        aqms_renamed[gb10y], 
        aqms_renamed[curr], 
        aqms_renamed[irf]
    ], axis=1)
    
    # Forward and backward fill missing values
    assets_imputed = assets.ffill().bfill()
    ir_imputed = ir.ffill().bfill()
    
    # Calculate annual returns for price-like assets
    price_cols = [col for col in assets.columns 
                 if col.startswith('Equity_') 
                 or col.startswith('Currency_') 
                 or 'IR Future' in col]
    
    log_returns = np.log(assets_imputed[price_cols] / assets_imputed[price_cols].shift(1))
    annual_log_returns = log_returns.resample('Y').sum()
    annual_returns = np.exp(annual_log_returns) - 1
    
    # Calculate annual average yields for bonds and interests
    annual_yields = assets_imputed[gb2y + gb10y].resample('Y').mean()
    annual_yields = annual_yields.diff() / 100
    ir_annual = ir_imputed.diff().resample('Y').mean()
    
    # Combine returns and yields
    annual_df = pd.concat([annual_returns, annual_yields], axis=1)
    
    return annual_df, ir_annual


def preprocess_macro_factors(bc_df, annual_df, ir):
    """
    Preprocess macro factors from business cycle data and asset returns.
    
    Args:
        bc_df (pd.DataFrame): Business cycle DataFrame
        annual_df (pd.DataFrame): Annual asset returns DataFrame
        ir: df of imputed and annual average dataframe
        
    Returns:
        pd.DataFrame: Processed macro factors DataFrame
    """
    # Initialize macro factors DataFrame
    mf = pd.DataFrame(index=annual_df.index)
    
    # Country code mapping for consistent naming
    country_to_abbr = {
        'US': 'US',
        'United States': 'US',
        'UK': 'UK',
        'United Kingdom': 'UK',
        'Japan': 'JP',
        'Hong Kong': 'HK',
        'Hong Kong SAR': 'HK',
        'Canada': 'CA',
        'Euro': 'EU',
        'European Union': 'EU',
        'Switzerland': 'CH',
        'Australia': 'AU',
        'New Zealand': 'NZ',
        'Switzerland': 'CH'
    }
    
    # Rename columns in business cycle data
    renamed_columns = {}
    for col in bc_df.columns:
        if col.startswith('GDP_'):
            for country, abbr in country_to_abbr.items():
                if col == f'GDP_{country}':
                    renamed_columns[col] = f'GDP_{abbr}'
        elif col.startswith('CPI_'):
            for country, abbr in country_to_abbr.items():
                if col == f'CPI_{country}':
                    renamed_columns[col] = f'CPI_{abbr}'
    
    bc_renamed = bc_df.rename(columns=renamed_columns)
    bc_renamed = bc_renamed[renamed_columns.values()]

    # Convert business cycle data to numeric, handling 'no data' entries
    for col in bc_renamed.columns:
        if col.startswith(('GDP_', 'CPI_')):
            # Convert to numeric, coercing errors to NaN
            bc_renamed[col] = pd.to_numeric(bc_renamed[col], errors='coerce')
            # Divide by 100 only for valid numeric values
            bc_renamed[col] = bc_renamed[col] / 100
    
    # Calculate excess returns (equity returns minus risk-free rate)
    countries = ['US', 'UK', 'Japan', 'Hong Kong', 'Canada', 'Euro', 'Australia', 'New Zealand','Switzerland']
    
    # Calculate excess returns for each country
    for country in countries:
        equity_return = annual_df[f'Equity_{country}']
        risk_free_rate = ir[f'Interest Rates_{country_to_abbr[country]}']
        mf[f'Excess_Return_{country_to_abbr[country]}'] = equity_return - risk_free_rate
    
    # Add business cycle data (GDP and CPI)
    mf = mf.join(bc_renamed, how='left')
    
    # Add currency returns
    currency_cols = [col for col in annual_df.columns if col.startswith("Currency_")]
    mf = mf.join(annual_df[currency_cols], how='left')
    
    # Add monetary policy proxies (2-year bond yields)
    monetary_policy_cols = [col for col in annual_df.columns 
                          if col.startswith("Bond Yield 2Y_")]
    mf = mf.join(annual_df[monetary_policy_cols], how='left')
    
    # Filter to desired time period (1980-2025)
    mf = mf[(mf.index.year >= 1980) & (mf.index.year <= 2025)]
    
    return mf

def preprocess_asset_data_daily(aqms_df):
    """
    Preprocess asset data to calculate annual returns and yields.
    
    Args:
        aqms_df (pd.DataFrame): Raw AQMS DataFrame
        
    Returns:
        pd.DataFrame: Processed annual returns and yields
    """
    # Define asset categories
    equity = ['US', 'UK', 'Japan', 'Hong Kong', 'Canada', 'Euro', 'Switzerland', 'New Zealand', 'Australia']
    
    int_future = ['SFRA Comdty', 'SFR1YZ2 Comdty', 'SFR1YZ3 Comdty', 
                  'SFR1YZ4 Comdty', 'SFR1YZ5 Comdty', 'SFR1YZ6 Comdty', 
                  'SFR2YZ2 Comdty']
    
    raw = ['JP', 'EU', 'HK', 'CH', 'CA', 'AU', 'NZ']
    currency = raw + ['GBP']
    gov_bond = raw + ['US', 'UK']
    
    # Initialize lists for asset columns
    gb2y, gb10y, curr, eqt, irf = [], [], [], [], []
    
    # Categorize columns
    for name in aqms_df.columns:
        for e in equity:
            if (e in name) and ('Equity' in name) and ('Pan' not in name):
                eqt.append(name)
                
        for g in gov_bond:
            if (g in name) and ('2' in name) and ('Bond' in name):
                gb2y.append(name)
            if (g in name) and ('10' in name) and ('Bond' in name):
                gb10y.append(name)
                
        for c in currency:
            if (c in name) and ('Curncy' in name):
                curr.append(name)
                
        for i in int_future:
            if (i in name) and ('Future' in name):
                irf.append(name)
    
    # Process interest rate columns
    rate_mapping = {
        'US': ['FDTR Index'],
        'UK': ['UKBRBASE Index'],
        'JP': ['BOJDTR Index'],
        'EU': ['EURR002W Index'],
        'HK': ['PRIMHK Index'],
        'CA': ['CABROVER Index'],
        'AU': ['RBATCTR Index'],
        'NZ': ['NZOCR Index'],
        'CH': ['SZNBPOLR Index']
    }
    
    # Create renaming dictionary
    renaming_dict = {}
    for col in aqms_df.columns:
        for abbr, indices in rate_mapping.items():
            for index in indices:
                if index in col:
                    renaming_dict[col] = col.replace(index, abbr)
    
    # Rename columns and get interest rate columns
    aqms_renamed = aqms_df.rename(columns=renaming_dict)
    ir = aqms_renamed[list(renaming_dict.values())]
    
    # Combine all asset data
    assets = pd.concat([
        aqms_renamed[eqt], 
        aqms_renamed[gb2y], 
        aqms_renamed[gb10y], 
        aqms_renamed[curr], 
        aqms_renamed[irf]
    ], axis=1)
    
    # Forward and backward fill missing values
    assets_imputed = assets.ffill().bfill()
    ir_imputed = ir.ffill().bfill()
    
    # Calculate annual returns for price-like assets
    price_cols = [col for col in assets.columns 
                 if col.startswith('Equity_') 
                 or col.startswith('Currency_') 
                 or 'IR Future' in col]
    
    # log_returns = np.log(assets_imputed[price_cols] / assets_imputed[price_cols].shift(1))
    # annual_log_returns = log_returns.resample('Y').sum()
    # annual_returns = np.exp(annual_log_returns) - 1
    daily_returns = assets_imputed[price_cols].pct_change()

    # Calculate annual average yields for bonds and interests
    daily_yields = assets_imputed[gb2y + gb10y].diff() / 100
    ir_annual = ir_imputed.diff() / 100
    
    # Combine returns and yields
    daily_df = pd.concat([daily_returns, daily_yields], axis=1)
    
    return daily_df, ir_imputed



def preprocess_macro_factors(bc_df, annual_df, ir):
    """
    Preprocess macro factors from business cycle data and asset returns.
    
    Args:
        bc_df (pd.DataFrame): Business cycle DataFrame
        annual_df (pd.DataFrame): Annual asset returns DataFrame
        ir: df of imputed and annual average dataframe
        
    Returns:
        pd.DataFrame: Processed macro factors DataFrame
    """
    # Initialize macro factors DataFrame
    mf = pd.DataFrame(index=annual_df.index)
    
    # Country code mapping for consistent naming
    country_to_abbr = {
        'US': 'US',
        'United States': 'US',
        'UK': 'UK',
        'United Kingdom': 'UK',
        'Japan': 'JP',
        'Hong Kong': 'HK',
        'Hong Kong SAR': 'HK',
        'Canada': 'CA',
        'Euro': 'EU',
        'European Union': 'EU',
        'Switzerland': 'CH',
        'Australia': 'AU',
        'New Zealand': 'NZ',
        'Switzerland': 'CH'
    }
    
    # Rename columns in business cycle data
    renamed_columns = {}
    for col in bc_df.columns:
        if col.startswith('GDP_'):
            for country, abbr in country_to_abbr.items():
                if col == f'GDP_{country}':
                    renamed_columns[col] = f'GDP_{abbr}'
        elif col.startswith('CPI_'):
            for country, abbr in country_to_abbr.items():
                if col == f'CPI_{country}':
                    renamed_columns[col] = f'CPI_{abbr}'
    
    bc_renamed = bc_df.rename(columns=renamed_columns)
    bc_renamed = bc_renamed[renamed_columns.values()]

    # Convert business cycle data to numeric, handling 'no data' entries
    for col in bc_renamed.columns:
        if col.startswith(('GDP_', 'CPI_')):
            # Convert to numeric, coercing errors to NaN
            bc_renamed[col] = pd.to_numeric(bc_renamed[col], errors='coerce')
            # Divide by 100 only for valid numeric values
            bc_renamed[col] = bc_renamed[col] / 100
    
    # Calculate excess returns (equity returns minus risk-free rate)
    countries = ['US', 'UK', 'Japan', 'Hong Kong', 'Canada', 'Euro', 'Australia', 'New Zealand','Switzerland']
    
    # Calculate excess returns for each country
    for country in countries:
        equity_return = annual_df[f'Equity_{country}']
        risk_free_rate = ir[f'Interest Rates_{country_to_abbr[country]}']
        mf[f'Excess_Return_{country_to_abbr[country]}'] = equity_return - risk_free_rate
    
    # Add business cycle data (GDP and CPI)
    mf = mf.join(bc_renamed, how='left')
    
    # Add currency returns
    currency_cols = [col for col in annual_df.columns if col.startswith("Currency_")]
    mf = mf.join(annual_df[currency_cols], how='left')
    
    # Add monetary policy proxies (2-year bond yields)
    monetary_policy_cols = [col for col in annual_df.columns 
                          if col.startswith("Bond Yield 2Y_")]
    mf = mf.join(annual_df[monetary_policy_cols], how='left')
    
    # Filter to desired time period (1980-2025)
    mf = mf[(mf.index.year >= 1980) & (mf.index.year <= 2025)]
    
    return mf

# Unify column names
def standardize_column_names(df):
    """
    Standardize column names to use consistent two-digit country abbreviations.
    
    Args:
        df (pd.DataFrame): DataFrame with columns to be renamed
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    # Country name to two-letter abbreviation mapping
    country_map = {
        'US': 'US',
        'UK': 'UK',
        'Japan': 'JP',
        'Hong Kong': 'HK',
        'Canada': 'CA',
        'Pan-Europe': 'EU',
        'Euro': 'EU',
        'Switzerland': 'CH',
        'Australia': 'AU',
        'New Zealand': 'NZ',
        'GBP': 'UK',  # Currency special cases
        'JPY': 'JP',
        'EUR': 'EU',
        'HKD': 'HK',
        'CHF': 'CH',
        'CAD': 'CA',
        'AUD': 'AU',
        'NZD': 'NZ'
    }
    
    new_columns = []
    for col in df.columns:
        parts = col.split('_')
        
        # Handle different column patterns
        if col.startswith('Equity_'):
            country = parts[1]
            new_col = f"Equity_{country_map.get(country, country)}"
            
        elif col.startswith('Currency_'):
            currency = parts[1].split()[0]  # Get currency code before "Curncy"
            new_col = f"Currency_{country_map.get(currency, currency)}"
            
        elif col.startswith('IR Future_'):
            # Keep IR Future columns as-is (they're contracts, not countries)
            new_col = col
            
        elif col.startswith(('Bond Yield 2Y_', 'Bond Yield 10Y_')):
            # Extract country code from bond yield columns
            bond_part = parts[1]
            if bond_part.startswith('GT'):  # Handle GT-prefixed bonds (e.g., GTJPY)
                country_code = bond_part[2:4]  # Gets JP from GTJPY
            elif bond_part.startswith('GU'):  # Handle UK bonds (GUKG)
                country_code = 'UK'
            elif bond_part.startswith('US'):  # US bonds
                country_code = 'US'
            elif bond_part.startswith('HK'):  # Hong Kong bonds
                country_code = 'HK'
            else:
                # Fallback - take first 2 characters
                country_code = bond_part[:2]
            new_col = f"{parts[0].replace(' ','')}_{country_code}"
    
        elif col.startswith('Excess_Return_'):
            country = parts[2]  # Changed from parts[1] to parts[2]
            new_col = f"ExcessReturn_{country_map.get(country, country)}"
            
        elif col.startswith(('GDP_', 'CPI_')):
            country = parts[1]
            new_col = f"{parts[0]}_{country_map.get(country, country)}"
            
        else:
            new_col = col  # Leave unchanged if no pattern matches
            
        new_columns.append(new_col)
    
    # Apply the new column names
    renamed_df = df.copy()
    renamed_df.columns = new_columns
    
    return renamed_df
def process_ir_futures(ar_df):
    """
    Process interest rate futures data by:
    1. Dropping existing IR Future columns
    2. Generating pseudo-data for each country
    
    Args:
        ar_df (pd.DataFrame): Asset returns DataFrame
        
    Returns:
        pd.DataFrame: Processed DataFrame with pseudo IR Futures
    """
    # 1. Drop all existing IR Future columns
    ir_future_cols = [col for col in ar_df.columns if 'IR Future' in col]
    ar_processed = ar_df.drop(columns=ir_future_cols)
    
    # 2. Generate pseudo-data for each country
    countries = ['US', 'UK', 'JP', 'HK', 'CA', 'EU', 'CH', 'AU', 'NZ']
    
    for country in countries:
        col_name = f'IRFutures_{country}'
        
        # Generate random returns between -0.02 and 0.02 (2%)
        pseudo_data = np.random.uniform(low=-0.02, high=0.02, size=len(ar_df))
        
        # Add slight autocorrelation to make it more realistic
        for i in range(1, len(pseudo_data)):
            pseudo_data[i] = 0.7*pseudo_data[i-1] + 0.3*pseudo_data[i]
        
        ar_processed[col_name] = pseudo_data
    
    return ar_processed
def transform_currency_returns(ar_df):
    """
    Transform currency columns into asset returns in ar DataFrame.
    Handles JPY special case (already USD/JPY) and converts others to USD-based returns.
    
    Args:
        ar_df (pd.DataFrame): Asset returns DataFrame
        
    Returns:
        pd.DataFrame: Transformed DataFrame with currency returns
    """
    # # Currency columns to process (excluding JPY)
    # currency_cols = [col for col in ar_df.columns 
    #                if col.startswith('Currency_') and not col.endswith('JP')]
    
    # # JPY column (special handling)
    # jpy_col = [col for col in ar_df.columns if col.endswith('JP')][0]
    
    # # Transform non-JPY currencies (currently USD/FCY → need FCY/USD)
    # for col in currency_cols:
    #     # Convert from USD/FCY to FCY/USD and calculate returns
    #     ar_df[col] = (1 / ar_df[col]).pct_change()
    
    # # Transform JPY (currently USD/JPY → keep as is for returns)
    # ar_df[jpy_col] = ar_df[jpy_col].pct_change()

    jpy_col = 'Currency_JP'
    ar_df[jpy_col] = - ar_df[jpy_col]
    
    # Rename columns to reflect they're now returns
    ar_df.columns = [col.replace('Currency_', 'FXReturn_') for col in ar_df.columns]
    
    return ar_df


def create_us_centric_trade_factors(mf_df):
    """
    Create trade factors assuming each country primarily trades with the US.
    For US, creates an equally-weighted basket of all other currencies.
    
    Args:
        mf_df (pd.DataFrame): Macro factors DataFrame
        
    Returns:
        pd.DataFrame: Updated DataFrame with trade factors for all countries
    """
    # Get all currency columns (USD per FCY)
    currency_cols = [col for col in mf_df.columns if col.startswith('Currency_')]
    countries = [col.split('_')[1] for col in currency_cols]
    
    # Create trade factors for each country
    for country in countries + ['US']:  # Include US separately
        if country == 'US':
            # For US: equally-weighted basket of all other currencies
            changes = []
            for col in currency_cols:
                other_country = col.split('_')[1]
                
                if other_country == 'JP':
                    changes.append(1/(1+mf_df[col])-1)  # JPY is USD/JPY
                else:
                    changes.append(mf_df[col])  # Others are FCY/USD
            if changes:
                mf_df[f'TradeFactor_US'] = pd.DataFrame(changes).mean()
        else:
            # For non-US countries: use their currency vs USD
            col = f'Currency_{country}'
            if country == 'JP':
                mf_df[f'TradeFactor_{country}'] = mf_df[col]
            else:
                mf_df[f'TradeFactor_{country}'] = mf_df[col]
    
    # Apply 1-year smoothing
    # trade_cols = [col for col in mf_df.columns if col.startswith('TradeFactor_')]
    # mf_df[trade_cols] = mf_df[trade_cols].rolling(window=12).mean()
    
    return mf_df
def standardize_weights(raw_weights):
    """
    More robust weight standardization with debugging
    """
    standardized = pd.DataFrame(index=raw_weights.index, columns=raw_weights.columns)
    
    for date, row in raw_weights.iterrows():
        if row.isna().all():
            continue
            
        # Calculate z-scores with minimum divisor
        row_std = row.std()
        divisor = row_std if row_std > 1e-8 else 1.0  # Prevent divide-by-zero
        z_scores = (row - row.mean()) / divisor
        
        # Initialize weights
        weights = pd.Series(0, index=z_scores.index)
        
        # Long positions (positive z-scores)
        long_mask = z_scores > 0
        if long_mask.any():
            long_weights = z_scores[long_mask]
            weights[long_mask] = long_weights / long_weights.sum()
        
        # Short positions (negative z-scores)
        short_mask = z_scores < 0
        if short_mask.any():
            short_weights = z_scores[short_mask]
            weights[short_mask] = short_weights / (-short_weights.sum())
        
        standardized.loc[date] = weights.values
    
    return standardized


def calculate_bc_momentum(mf_df):
    """
    Calculate Business Cycle momentum scores for each country using:
    - 50% 1-year GDP growth change
    - 50% 1-year CPI inflation change
    """
    # Calculate 1-year changes for GDP and CPI
    gdp_changes = mf_df.filter(like='GDP_').diff()
    cpi_changes = mf_df.filter(like='CPI_').diff()
    
    # Combine 50/50 with proper sign conventions
    bc_scores = {}
    for country in [col.split('_')[1] for col in mf_df.columns if col.startswith('GDP_')]:
        gdp_col = f'GDP_{country}'
        cpi_col = f'CPI_{country}'
        
        # GDP: Higher growth = positive signal
        gdp_signal = gdp_changes[gdp_col]
        
        # CPI: Higher inflation = negative signal (except for currencies)
        cpi_signal = -cpi_changes[cpi_col]
        
        # Combine 50/50
        bc_scores[country] = 0.5*gdp_signal + 0.5*cpi_signal
    
    return pd.DataFrame(bc_scores)



def construct_bc_portfolio(ar_df, mf_df, target_vol = 0.05):
    """
    Construct complete Business Cycle long-short portfolio across all asset classes
    """
    # Calculate BC momentum scores
    bc_scores = calculate_bc_momentum(mf_df)
    
    # Create portfolios for each asset class
    equity_weights = create_asset_class_portfolio_bc(ar_df, bc_scores, 'Equity',target_vol=target_vol)
    fx_weights = create_asset_class_portfolio_bc(ar_df, bc_scores, 'FX',target_vol=target_vol)
    #bond2y_weights = create_asset_class_portfolio_bc(ar_df, bc_scores, 'Bond2Y',target_vol=target_vol)
    bond10y_weights = create_asset_class_portfolio_bc(ar_df, bc_scores, 'Bond10Y',target_vol=target_vol)
    # ir_weights = create_asset_class_portfolio(ar_df, bc_scores, 'IRFutures',target_vol=target_vol)
    
    # Combine all weights
    all_weights = pd.concat([
        equity_weights,
        fx_weights,
        #bond2y_weights/2,
        bond10y_weights,
        # ir_weights
    ], axis=1).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (all_weights.shift(1) * ar_df[all_weights.columns]).sum(axis=1)
    
    return {
        'weights': all_weights,
        'returns': portfolio_returns,
        'components': {
            'Equity': equity_weights,
            'FX': fx_weights,
            #'Bond2Y': bond2y_weights,
            'Bond10Y': bond10y_weights,
            # 'IRFutures': ir_weights
        }
    }
def calculate_it_momentum(mf_df):
    # It's no longer lagged like GDP
    it_factors = mf_df.filter(like='TradeFactor_')
    
    it_scores = {}
    for country in [col.split('_')[1] for col in mf_df.columns if col.startswith('TradeFactor_')]:
        it_col = f'TradeFactor_{country}'
        
        # Here sign is positive, control later
        it_signal = it_factors[it_col]
        it_scores[country] = -it_signal # Turn appreciation into deprecition
        
    return pd.DataFrame(it_scores)



def construct_it_portfolio(ar_df, mf_df,target_vol=0.05):
    """
    Construct complete International Trade long-short portfolio across all asset classes
    """
    # Calculate BC momentum scores
    it_scores = calculate_it_momentum(mf_df)
    
    # Create portfolios for each asset class
    equity_weights = create_asset_class_portfolio_it(ar_df, it_scores, 'Equity',target_vol=target_vol)
    fx_weights = create_asset_class_portfolio_it(ar_df, it_scores, 'FX',target_vol=target_vol)
    #bond2y_weights = create_asset_class_portfolio_it(ar_df, it_scores, 'Bond2Y',target_vol=target_vol)
    bond10y_weights = create_asset_class_portfolio_it(ar_df, it_scores, 'Bond10Y',target_vol=target_vol)
    # ir_weights = create_asset_class_portfolio(ar_df, it_scores, 'IRFutures',target_vol=target_vol)
    
    # Combine all weights
    all_weights = pd.concat([
        equity_weights,
        fx_weights,
        #bond2y_weights/2,
        bond10y_weights,
        # ir_weights
    ], axis=1).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (all_weights.shift(1) * ar_df[all_weights.columns]).sum(axis=1)
    
    return {
        'weights': all_weights,
        'returns': portfolio_returns,
        'components': {
            'Equity': equity_weights,
            'FX': fx_weights,
            #'Bond2Y': bond2y_weights,
            'Bond10Y': bond10y_weights,
            # 'IRFutures': ir_weights
        }
    }
def calculate_mp_momentum(mf_df):
    # It's no longer lagged like GDP
    mp_factors = mf_df.filter(like='BondYield2Y_')
    
    mp_scores = {}
    for country in [col.split('_')[1] for col in mf_df.columns if col.startswith('BondYield2Y_')]:
        mp_col = f'BondYield2Y_{country}'
        
        # Here sign is positive, control later
        mp_signal = mp_factors[mp_col].diff() # Here signal is the change in yield so diffed
        mp_scores[country] = mp_signal
        # print(mp_signal)
    
    return pd.DataFrame(mp_scores)




def construct_mp_portfolio(ar_df, mf_df,target_vol=0.05):
    """
    Construct complete Business Cycle long-short portfolio across all asset classes
    """
    # Calculate BC momentum scores
    mp_scores = calculate_mp_momentum(mf_df)
    
    # Create portfolios for each asset class
    equity_weights = create_asset_class_portfolio_mp(ar_df, mp_scores, 'Equity',target_vol=target_vol)
    fx_weights = create_asset_class_portfolio_mp(ar_df, mp_scores, 'FX',target_vol=target_vol)
    #bond2y_weights = create_asset_class_portfolio_mp(ar_df, mp_scores, 'Bond2Y')
    bond10y_weights = create_asset_class_portfolio_mp(ar_df, mp_scores, 'Bond10Y',target_vol=target_vol)
    # ir_weights = create_asset_class_portfolio(ar_df, mp_scores, 'IRFutures')
    
    # Combine all weights
    all_weights = pd.concat([
        equity_weights,
        fx_weights,
        #bond2y_weights/2,
        bond10y_weights,
        # ir_weights
    ], axis=1).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (all_weights.shift(1) * ar_df[all_weights.columns]).sum(axis=1)
    
    return {
        'weights': all_weights,
        'returns': portfolio_returns,
        'components': {
            'Equity': equity_weights,
            'FX': fx_weights,
            #'Bond2Y': bond2y_weights,
            'Bond10Y': bond10y_weights,
            # 'IRFutures': ir_weights
        }
    }
def calculate_rs_momentum(mf_df):
    # It's no longer lagged like GDP
    rs_factors = mf_df.filter(like='ExcessReturn_')
    
    rs_scores = {}
    for country in [col.split('_')[1] for col in mf_df.columns if col.startswith('ExcessReturn_')]:
        rs_col = f'ExcessReturn_{country}'
        
        # Here sign is positive, control later
        rs_signal = rs_factors[rs_col].diff() # Here signal is the change in yield so diffed
        rs_scores[country] = rs_signal
    
    return pd.DataFrame(rs_scores)





def construct_rs_portfolio(ar_df, mf_df, target_vol=0.05):
    """
    Construct complete Business Cycle long-short portfolio across all asset classes
    """
    # Calculate BC momentum scores
    rs_scores = calculate_rs_momentum(mf_df)
    
    # Create portfolios for each asset class
    equity_weights = create_asset_class_portfolio_rs(ar_df, rs_scores, 'Equity',target_vol=target_vol)
    fx_weights = create_asset_class_portfolio_rs(ar_df, rs_scores, 'FX',target_vol=target_vol)
    # bond2y_weights = create_asset_class_portfolio_rs(ar_df, rs_scores, 'Bond2Y')
    bond10y_weights = create_asset_class_portfolio_rs(ar_df, rs_scores, 'Bond10Y',target_vol=target_vol)
    # ir_weights = create_asset_class_portfolio(ar_df, rs_scores, 'IRFutures')
    
    # Combine all weights
    all_weights = pd.concat([
        equity_weights,
        fx_weights,
        #bond2y_weights/2,
        bond10y_weights,
        # ir_weights
    ], axis=1).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (all_weights.shift(1) * ar_df[all_weights.columns]).sum(axis=1)
    
    return {
        'weights': all_weights,
        'returns': portfolio_returns,
        'components': {
            'Equity': equity_weights,
            'FX': fx_weights,
            # 'Bond2Y': bond2y_weights,
            'Bond10Y': bond10y_weights,
            # 'IRFutures': ir_weights
        }
    }


def create_asset_class_portfolio_mp(asset_returns, mp_scores, asset_class, target_vol=0.01):
    """
    Modified version with correct weight standardization
    """
    # Get relevant assets for this class
    if asset_class == 'Equity':
        assets = [col for col in asset_returns.columns if col.startswith('Equity_')]
        countries = [col.split('_')[1] for col in assets]
        # Equities: Yield up -> stock val down -> bearish to equity
        raw_scores = -mp_scores[countries]  # Get scores for relevant countries
        
    elif asset_class == 'FX':
        assets = [col for col in asset_returns.columns if col.startswith('FXReturn_')]
        countries = [col.split('_')[1] for col in assets]
        # FX: Yield up -> bullish for currency
        raw_scores = mp_scores[countries]
        
    elif asset_class in ['Bond2Y', 'Bond10Y', 'IRFutures']:
        prefix = {
            'Bond2Y': 'BondYield2Y_',
            'Bond10Y': 'BondYield10Y_',
            'IRFutures': 'IRFutures_'
        }[asset_class]
        assets = [col for col in asset_returns.columns if col.startswith(prefix)]
        countries = [col.split('_')[-1] for col in assets]
        # Fixed Income: Yield up -> price for fi assets down -> bullish for fi
        raw_scores = -mp_scores[countries]
        
    # Create DataFrame of raw scores aligned wmph asset returns index
    raw_weights = pd.DataFrame(index=asset_returns.index, columns=assets)
    for asset, country in zip(assets, countries):
        raw_weights[asset] = raw_scores[country]
    
    # Standardize weights using row-wise z-scoring
    weights = standardize_weights(raw_weights)
    
    if len(assets) > 1:
        returns = asset_returns[assets].dropna()
        weights = weights.loc[returns.index]
        min_years = 5
        portfolio_vol = pd.Series(index=weights.index, dtype=float)
        
        for i in range(min_years, len(weights)):
            current_date = weights.index[i]
            lookback_dates = weights.index[i-min_years:i]
            w_current = weights.loc[current_date].values.reshape(-1, 1)
            hist_returns = returns.loc[lookback_dates]
            
            if len(hist_returns.dropna()) >= min_years:
                C = hist_returns.cov()
                
                # Convert covariance to numpy array if needed
                C_matrix = C.values if hasattr(C, 'values') else C
                
                try:
                    # Proper matrix multiplication and scalar conversion
                    var = float(w_current.T @ C_matrix @ w_current)
                    portfolio_vol.loc[current_date] = np.sqrt(var)
                except Exception as e:
                    print(f"Vol calc error at {current_date}: {str(e)}")
                    portfolio_vol.loc[current_date] = np.nan
        
        # Forward fill and apply scaling
        portfolio_vol = portfolio_vol.ffill()
        scaling_factors = target_vol / np.sqrt(portfolio_vol.replace(0, np.nan))
        weights = weights.mul(scaling_factors, axis=0)
    
    return weights


def create_asset_class_portfolio_it(asset_returns, it_scores, asset_class, target_vol=0.01):
    """
    Modified version with correct weight standardization
    """
    # Get relevant assets for this class
    if asset_class == 'Equity':
        assets = [col for col in asset_returns.columns if col.startswith('Equity_')]
        countries = [col.split('_')[1] for col in assets]
        # Equities: Depre -> High trade -> good for equity
        raw_scores = -it_scores[countries]  # Get scores for relevant countries
        
    elif asset_class == 'FX':
        assets = [col for col in asset_returns.columns if col.startswith('FXReturn_')]
        countries = [col.split('_')[1] for col in assets]
        # FX: Depre -> bearish for fx
        raw_scores = it_scores[countries]
        
    elif asset_class in ['Bond2Y', 'Bond10Y', 'IRFutures']:
        prefix = {
            'Bond2Y': 'BondYield2Y_',
            'Bond10Y': 'BondYield10Y_',
            'IRFutures': 'IRFutures_'
        }[asset_class]
        assets = [col for col in asset_returns.columns if col.startswith(prefix)]
        countries = [col.split('_')[-1] for col in assets]
        # Fixed Income: Depre -> High trade -> Reduce inflation&economy -> possible rate decrease -> bullish for Fixed Income
        raw_scores = it_scores[countries]
        
    # Create DataFrame of raw scores aligned with asset returns index
    raw_weights = pd.DataFrame(index=asset_returns.index, columns=assets)
    for asset, country in zip(assets, countries):
        raw_weights[asset] = raw_scores[country]
    
    # Standardize weights using row-wise z-scoring
    weights = standardize_weights(raw_weights)
    
    if len(assets) > 1:
        returns = asset_returns[assets].dropna()
        weights = weights.loc[returns.index]
        min_years = 5
        portfolio_vol = pd.Series(index=weights.index, dtype=float)
        
        for i in range(min_years, len(weights)):
            current_date = weights.index[i]
            lookback_dates = weights.index[i-min_years:i]
            w_current = weights.loc[current_date].values.reshape(-1, 1)
            hist_returns = returns.loc[lookback_dates]
            
            if len(hist_returns.dropna()) >= min_years:
                C = hist_returns.cov()
                
                # Convert covariance to numpy array if needed
                C_matrix = C.values if hasattr(C, 'values') else C
                
                try:
                    # Proper matrix multiplication and scalar conversion
                    var = float(w_current.T @ C_matrix @ w_current)
                    portfolio_vol.loc[current_date] = np.sqrt(var)
                except Exception as e:
                    print(f"Vol calc error at {current_date}: {str(e)}")
                    portfolio_vol.loc[current_date] = np.nan
        
        # Forward fill and apply scaling
        portfolio_vol = portfolio_vol.ffill()
        scaling_factors = target_vol / np.sqrt(portfolio_vol.replace(0, np.nan))
        weights = weights.mul(scaling_factors, axis=0)
    
    return weights


def create_asset_class_portfolio_bc(asset_returns, bc_scores, asset_class, target_vol=0.01):
    """
    Modified version with correct weight standardization
    """
    # Get relevant assets for this class
    if asset_class == 'Equity':
        assets = [col for col in asset_returns.columns if col.startswith('Equity_')]
        countries = [col.split('_')[1] for col in assets]
        # Equities: Long growth/inflation decline
        raw_scores = bc_scores[countries]  # Get scores for relevant countries
        
    elif asset_class == 'FX':
        assets = [col for col in asset_returns.columns if col.startswith('FXReturn_')]
        countries = [col.split('_')[1] for col in assets]
        # FX: Long growth/inflation increase (Balassa-Samuelson)
        raw_scores = bc_scores[countries]
        
    elif asset_class in ['Bond2Y', 'Bond10Y', 'IRFutures']:
        prefix = {
            'Bond2Y': 'BondYield2Y_',
            'Bond10Y': 'BondYield10Y_',
            'IRFutures': 'IRFutures_'
        }[asset_class]
        assets = [col for col in asset_returns.columns if col.startswith(prefix)]
        countries = [col.split('_')[-1] for col in assets]
        # Fixed Income: Short growth/inflation increase
        raw_scores = -bc_scores[countries]
        
    # Create DataFrame of raw scores aligned with asset returns index
    raw_weights = pd.DataFrame(index=asset_returns.index, columns=assets)
    for asset, country in zip(assets, countries):
        raw_weights[asset] = raw_scores[country]
    
    # Standardize weights using row-wise z-scoring
    weights = standardize_weights(raw_weights)
    
    if len(assets) > 1:
        returns = asset_returns[assets].dropna()
        weights = weights.loc[returns.index]
        min_years = 5
        portfolio_vol = pd.Series(index=weights.index, dtype=float)
        
        for i in range(min_years, len(weights)):
            current_date = weights.index[i]
            lookback_dates = weights.index[i-min_years:i]
            w_current = weights.loc[current_date].values.reshape(-1, 1)
            hist_returns = returns.loc[lookback_dates]
            
            if len(hist_returns.dropna()) >= min_years:
                C = hist_returns.cov()
                
                # Convert covariance to numpy array if needed
                C_matrix = C.values if hasattr(C, 'values') else C
                
                try:
                    # Proper matrix multiplication and scalar conversion
                    var = float(w_current.T @ C_matrix @ w_current)
                    portfolio_vol.loc[current_date] = np.sqrt(var)
                except Exception as e:
                    print(f"Vol calc error at {current_date}: {str(e)}")
                    portfolio_vol.loc[current_date] = np.nan
        
        # Forward fill and apply scaling
        portfolio_vol = portfolio_vol.ffill()
        scaling_factors = target_vol / np.sqrt(portfolio_vol.replace(0, np.nan))
        weights = weights.mul(scaling_factors, axis=0)
    
    return weights

def create_asset_class_portfolio_rs(asset_returns, rs_scores, asset_class, target_vol=0.01):
    """
    Modified version with correct weight standardization
    """
    # Get relevant assets for this class
    if asset_class == 'Equity':
        assets = [col for col in asset_returns.columns if col.startswith('Equity_')]
        countries = [col.split('_')[1] for col in assets]
        # Equities: Yield up -> stock val down -> bearish to equity
        raw_scores = -rs_scores[countries]  # Get scores for relevant countries
        
    elif asset_class == 'FX':
        assets = [col for col in asset_returns.columns if col.startswith('FXReturn_')]
        countries = [col.split('_')[1] for col in assets]
        # FX: Yield up -> bullish for currency
        raw_scores = rs_scores[countries]
        
    elif asset_class in ['Bond2Y', 'Bond10Y', 'IRFutures']:
        prefix = {
            'Bond2Y': 'BondYield2Y_',
            'Bond10Y': 'BondYield10Y_',
            'IRFutures': 'IRFutures_'
        }[asset_class]
        assets = [col for col in asset_returns.columns if col.startswith(prefix)]
        countries = [col.split('_')[-1] for col in assets]
        # Fixed Income: Yield up -> price for fi assets down -> bullish for fi
        raw_scores = -rs_scores[countries]
        
    # Create DataFrame of raw scores aligned with asset returns index
    raw_weights = pd.DataFrame(index=asset_returns.index, columns=assets)
    for asset, country in zip(assets, countries):
        raw_weights[asset] = raw_scores[country]
    
    # Standardize weights using row-wise z-scoring
    weights = standardize_weights(raw_weights)
    
    if len(assets) > 1:
        returns = asset_returns[assets].dropna()
        weights = weights.loc[returns.index]
        min_years = 5
        portfolio_vol = pd.Series(index=weights.index, dtype=float)
        
        for i in range(min_years, len(weights)):
            current_date = weights.index[i]
            lookback_dates = weights.index[i-min_years:i]
            w_current = weights.loc[current_date].values.reshape(-1, 1)
            hist_returns = returns.loc[lookback_dates]
            
            if len(hist_returns.dropna()) >= min_years:
                C = hist_returns.cov()
                
                # Convert covariance to numpy array if needed
                C_matrix = C.values if hasattr(C, 'values') else C
                
                try:
                    # Proper matrix multiplication and scalar conversion
                    var = float(w_current.T @ C_matrix @ w_current)
                    portfolio_vol.loc[current_date] = np.sqrt(var)
                except Exception as e:
                    print(f"Vol calc error at {current_date}: {str(e)}")
                    portfolio_vol.loc[current_date] = np.nan
        
        # Forward fill and apply scaling
        portfolio_vol = portfolio_vol.ffill()
        scaling_factors = target_vol / np.sqrt(portfolio_vol.replace(0, np.nan))
        weights = weights.mul(scaling_factors, axis=0)
    
    return weights