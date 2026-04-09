# 12h_1d_camarilla_breakout_v2
# Hypothesis: Use 12h timeframe with daily Camarilla levels (S4/R4) for breakout entries.
# Enter long when price breaks above R4 with volume confirmation; short when breaks below S4.
# Exit when price re-enters prior day's range. Uses 12h ATR for volatility filter.
# Designed for fewer trades (~20-40/year) to avoid fee drag, works in bull/bear via mean-reversion logic.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels (using prior day's OHLC)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (S4/R4) using prior day's OHLC
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    prev_high = np.full(len(df_d), np.nan)
    prev_low = np.full(len(df_d), np.nan)
    
    for i in range(1, len(df_d)):
        ph = df_d['high'].iloc[i-1]
        pl = df_d['low'].iloc[i-1]
        pc = df_d['close'].iloc[i-1]
        r4[i] = pc + (ph - pl) * 1.1 / 2  # R4
        s4[i] = pc - (ph - pl) * 1.1 / 2  # S4
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align daily values to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Load 12h data ONCE before loop for ATR(14) volatility filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h ATR(14)
    tr_12h = np.full(len(df_12h), np.nan)
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(1, len(df_12h)):
        high_low = df_12h['high'].iloc[i] - df_12h['low'].iloc[i]
        high_close = np.abs(df_12h['high'].iloc[i] - df_12h['close'].iloc[i-1])
        low_close = np.abs(df_12h['low'].iloc[i] - df_12h['close'].iloc[i-1])
        tr_12h[i] = max(high_low, high_close, low_close)
        if i >= 13:
            atr_12h[i] = np.mean(tr_12h[i-13:i+1])
    
    # Align 12h ATR to 12h timeframe (no change, but for consistency)
    atr_12h_aligned = atr_12h  # Already on 12h timeframe
    
    # Volume confirmation: 2-period average (2*12h = 1 day) to match daily timeframe
    vol_ma_2 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 2:
            vol_sum -= volume[i-2]
        if i >= 1:
            vol_ma_2[i] = vol_sum / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_2[i]) or 
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation and volatility filter
            vol_ratio = volume[i] / vol_ma_2[i] if vol_ma_2[i] > 0 else 0
            if (close[i] > r4_aligned[i] and 
                vol_ratio > 1.5 and 
                atr_12h_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation and volatility filter
            elif (close[i] < s4_aligned[i] and 
                  vol_ratio > 1.5 and 
                  atr_12h_aligned[i] > 0):
                position = -1
                signals[i] = -0.25
    
    return signals