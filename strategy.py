# 6h_1d_camarilla_breakout_v1
# Hypothesis: 6-hour breakouts above/below daily Camarilla pivot levels (H5/L5) with volume confirmation and ATR volatility filter.
# Uses H5/L5 levels (stronger breakout) for higher probability moves. Exits when price returns to daily pivot point (PP).
# Designed for 6h timeframe to reduce trade frequency vs 4h, targeting 50-150 total trades over 4 years.
# Works in both bull and bear markets as pivot levels adapt to volatility, and filters reduce whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.full(n, np.nan)
    if n >= 20:
        atr[19] = np.mean(tr[:20])
        for i in range(20, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Load 1d data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # H5 and L5 levels (stronger breakout levels: H5 = Close + 1.1*(H-L), L5 = Close - 1.1*(H-L))
    h5_1d = close_1d + (range_1d * 1.1)
    l5_1d = close_1d - (range_1d * 1.1)
    
    # Align 1d levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h5_aligned = align_htf_to_ltf(prices, df_1d, h5_1d)
    l5_aligned = align_htf_to_ltf(prices, df_1d, l5_1d)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Volume spike: current volume > 1.8x 20-period average (moderate filter)
    vol_ok = volume > vol_ma_20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.025 * close[i]  # ATR less than 2.5% of price
        
        if position == 1:  # Long position
            # Exit: price returns to or below Pivot Point
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Pivot Point
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H5 level with volume confirmation and volatility filter
            if close[i] > h5_aligned[i] and vol_ok[i] and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L5 level with volume confirmation and volatility filter
            elif close[i] < l5_aligned[i] and vol_ok[i] and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals