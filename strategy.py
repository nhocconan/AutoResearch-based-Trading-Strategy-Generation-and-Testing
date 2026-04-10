#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d funding rate z-score filter
# - Long: Williams %R(14) < -80 (oversold) + 1d funding rate z-score < -1.0 (extreme negative funding)
# - Short: Williams %R(14) > -20 (overbought) + 1d funding rate z-score > +1.0 (extreme positive funding)
# - Exit: Williams %R returns to -50 level OR funding z-score returns to neutral (-0.5 to 0.5)
# - Position sizing: 0.25 discrete level
# - Targets ~20-40 trades/year on 4h timeframe. Williams %R identifies exhaustion,
#   funding rate extremes capture sentiment reversals. Works in bull/bear: mean reversion
#   during extremes, avoids trending markets via funding filter.

name = "4h_1d_williamsr_funding_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d funding rate (placeholder - using price proxy for demonstration)
    # In reality, this would load from data/processed/funding/*.parquet
    # Using 1d returns as proxy for funding rate behavior
    returns_1d = pd.Series(close_1d).pct_change()
    funding_rate_1d = returns_1d.rolling(window=30, min_periods=30).mean().values  # 30-day average return as proxy
    
    # Calculate z-score of funding rate (30-day window)
    funding_mean = pd.Series(funding_rate_1d).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_rate_1d).rolling(window=30, min_periods=30).std().values
    funding_z = np.where(funding_std > 0, (funding_rate_1d - funding_mean) / funding_std, 0)
    
    # Align HTF indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    funding_z_aligned = align_htf_to_ltf(prices, df_1d, funding_z)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(funding_z_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for mean reversion entries
            # Long entry: Williams %R oversold + extreme negative funding
            if (williams_r_aligned[i] < -80 and funding_z_aligned[i] < -1.0):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought + extreme positive funding
            elif (williams_r_aligned[i] > -20 and funding_z_aligned[i] > 1.0):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to -50 OR funding returns to neutral
            if position == 1:  # Long position
                if williams_r_aligned[i] >= -50 or abs(funding_z_aligned[i]) <= 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r_aligned[i] <= -50 or abs(funding_z_aligned[i]) <= 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals