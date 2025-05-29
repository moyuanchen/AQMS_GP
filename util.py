import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
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
    annual_yields = assets_imputed[gb2y + gb10y].resample('Y').mean() / 100
    ir_annual = ir_imputed.resample('Y').mean() / 100
    
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
        'New Zealand': 'NZ'
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
    countries = ['US', 'UK', 'JP', 'Hong Kong', 'Canada', 'Euro', 'Australia', 'New Zealand']
    
    # Remove Switzerland (CH) as it's not in interest rate data
    if 'CH' in countries:
        countries.remove('CH')
    
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
    # Currency columns to process (excluding JPY)
    currency_cols = [col for col in ar_df.columns 
                   if col.startswith('Currency_') and not col.endswith('JP')]
    
    # JPY column (special handling)
    jpy_col = [col for col in ar_df.columns if col.endswith('JP')][0]
    
    # Transform non-JPY currencies (currently USD/FCY → need FCY/USD)
    for col in currency_cols:
        # Convert from USD/FCY to FCY/USD and calculate returns
        ar_df[col] = (1 / ar_df[col]).pct_change()
    
    # Transform JPY (currently USD/JPY → keep as is for returns)
    ar_df[jpy_col] = ar_df[jpy_col].pct_change()
    
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
                    changes.append(np.log(mf_df[col]).diff())  # JPY is USD/JPY
                else:
                    changes.append(np.log(1/mf_df[col]).diff())  # Others are FCY/USD
            if changes:
                mf_df[f'TradeFactor_US'] = pd.DataFrame(changes).mean()
        else:
            # For non-US countries: use their currency vs USD
            col = f'Currency_{country}'
            if country == 'JP':
                mf_df[f'TradeFactor_{country}'] = np.log(mf_df[col]).diff()
            else:
                mf_df[f'TradeFactor_{country}'] = np.log(1/mf_df[col]).diff()
    
    # Apply 1-year smoothing
    trade_cols = [col for col in mf_df.columns if col.startswith('TradeFactor_')]
    mf_df[trade_cols] = mf_df[trade_cols].rolling(window=12).mean()
    
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
    gdp_changes = mf_df.filter(like='GDP_').diff(12)
    cpi_changes = mf_df.filter(like='CPI_').diff(12)
    
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


def create_asset_class_portfolio(asset_returns, bc_scores, asset_class, target_vol=0.10):
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
        # Ensure we only use complete periods
        valid_returns = asset_returns[assets].dropna()
        
        # Initialize portfolio variance
        portfolio_variance = pd.Series(index=weights.index, dtype=float)
        
        for date in weights.index:
            if date in valid_returns.index:
                # Get current weights and filter to available assets
                current_weights = weights.loc[date]
                available_assets = current_weights.dropna().index
                
                if len(available_assets) > 0:
                    # Get corresponding covariance matrix
                    cov_matrix = valid_returns[available_assets].rolling(5).cov().loc[date]
                    
                    # Check dimension match
                    if cov_matrix.shape[0] == len(available_assets):
                        w = current_weights[available_assets].values.reshape(-1, 1)
                        C = cov_matrix.values
                        try:
                            var = (w.T @ C @ w).item()
                            portfolio_variance.loc[date] = var
                        except ValueError:
                            portfolio_variance.loc[date] = np.nan
                    else:
                        portfolio_variance.loc[date] = np.nan
                else:
                    portfolio_variance.loc[date] = np.nan
            else:
                portfolio_variance.loc[date] = np.nan
        
        # Calculate scaling factor
        scaling_factor = target_vol / np.sqrt(portfolio_variance)
        weights = weights.mul(scaling_factor, axis=0)
    
    return weights


def construct_bc_portfolio(ar_df, mf_df):
    """
    Construct complete Business Cycle long-short portfolio across all asset classes
    """
    # Calculate BC momentum scores
    bc_scores = calculate_bc_momentum(mf_df)
    
    # Create portfolios for each asset class
    equity_weights = create_asset_class_portfolio(ar_df, bc_scores, 'Equity')
    fx_weights = create_asset_class_portfolio(ar_df, bc_scores, 'FX')
    bond2y_weights = create_asset_class_portfolio(ar_df, bc_scores, 'Bond2Y')
    bond10y_weights = create_asset_class_portfolio(ar_df, bc_scores, 'Bond10Y')
    ir_weights = create_asset_class_portfolio(ar_df, bc_scores, 'IRFutures')
    
    # Combine all weights
    all_weights = pd.concat([
        equity_weights,
        fx_weights,
        bond2y_weights,
        bond10y_weights,
        ir_weights
    ], axis=1).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (all_weights.shift(1) * ar_df[all_weights.columns]).sum(axis=1)
    
    return {
        'weights': all_weights,
        'returns': portfolio_returns,
        'components': {
            'Equity': equity_weights,
            'FX': fx_weights,
            'Bond2Y': bond2y_weights,
            'Bond10Y': bond10y_weights,
            'IRFutures': ir_weights
        }
    }