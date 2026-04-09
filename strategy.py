#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter + volume spike confirmation
# - Primary signal: 6h Williams %R(14) crosses above -20 for short, below -80 for long
# - Trend filter: 1d EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 6h volume > 1.5 * 20-period EMA volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R captures mean reversion in ranges, EMA50 filter ensures
#   alignment with higher timeframe trend, reducing false signals in strong trends

name = "6h_1d_williams_r_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h volume regime: volume > 1.5 * 20-period EMA volume
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > (1.5 * volume_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 OR price crosses below EMA50
            if williams_r[i] > -20 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 OR price crosses above EMA50
            if williams_r[i] < -80 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume confirmation and EMA50 filter
            # Long: Williams %R crosses below -80 AND volume regime AND price above EMA50
            if williams_r[i] < -80 and volume_regime[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Williams %R crosses above -20 AND volume regime AND price below EMA50
            elif williams_r[i] > -20 and volume_regime[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals