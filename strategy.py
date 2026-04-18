# 12h_Pivot_R1_S1_Breakout_Volume_Confirmation_v1
# Hypothesis: On 12h chart, price breaking above R1 or below S1 daily pivot levels with volume confirmation captures breakouts with low whipsaw. Uses 1d pivot levels and volume spike filter to ensure institutional participation. Designed for ~25 trades/year to minimize fee drag and work in both bull and bear markets via breakout logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe (they update only after daily candle closes)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: >1.8x 20-period average on 12h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if price > r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike
            elif price < s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below pivot (mean reversion) or opposite breakout
            if price < pivot_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above pivot (mean reversion) or opposite breakout
            if price > pivot_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0