# 6h_Pivot_R3_S1_Breakout_VolumeSpike
# Uses 1d pivot points to identify strong support/resistance levels.
# Long when price breaks above S3 with volume confirmation, short when breaks below R3.
# Uses volume spike (2x 20-period average) to confirm institutional participation.
# Position size: 0.25. Target: 15-30 trades/year.
# Works in bull/bear: pivot levels act as dynamic support/resistance in all regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for pivot points and volume ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot points (standard floor method)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot_point - low_1d
    s1 = 2 * pivot_point - high_1d
    r2 = pivot_point + (high_1d - low_1d)
    s2 = pivot_point - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot_point - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot_point)
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike filter
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(volume_ma20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 6h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > (2.0 * volume_ma20_1d_aligned[i])
        
        if position == 0:
            # Long when price breaks above S3 with volume confirmation
            if close[i] > s3_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below R3 with volume confirmation
            elif close[i] < r3_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R3_S1_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0