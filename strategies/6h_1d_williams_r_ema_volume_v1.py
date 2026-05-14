#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA filter + volume confirmation
# - Primary signal: Williams %R(14) on 6h timeframe - oversold (< -80) for long, overbought (> -20) for short
# - Trend filter: 1d EMA50 - price must be above EMA for longs, below for shorts (higher timeframe alignment)
# - Volume confirmation: 6h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Williams %R captures mean reversions in ranges, EMA50 filter ensures alignment with higher timeframe trend

name = "6h_1d_williams_r_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute Williams %R on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close_6h) / (highest_high - lowest_low),
                          -50)  # neutral when no range
    
    # 6h volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum weakening) OR price crosses below 1d EMA50
            if williams_r[i] > -50 or close_6h[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum weakening) OR price crosses above 1d EMA50
            if williams_r[i] < -50 or close_6h[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation and 1d EMA50 filter
            # Long: Williams %R < -80 (oversold) AND volume regime AND price above 1d EMA50
            if (williams_r[i] < -80 and 
                volume_regime[i] and 
                close_6h[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND volume regime AND price below 1d EMA50
            elif (williams_r[i] > -20 and 
                  volume_regime[i] and 
                  close_6h[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals