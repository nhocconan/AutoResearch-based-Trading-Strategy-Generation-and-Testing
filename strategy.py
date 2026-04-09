#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_v10
# Hypothesis: Uses Camarilla pivot levels from 1d timeframe on 4h chart with volume confirmation and ATR stoploss.
# Long when price crosses above L4 (support) with volume > 1.3x average; short when price crosses below H4 (resistance) with volume > 1.3x average.
# Includes volatility filter using ATR to avoid choppy markets. Designed to work in both bull and bear markets by fading overextensions at key levels.
# Target: 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v10"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day
    # Formula: based on previous day's high, low, close
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla levels
    # H4 = close + 1.5 * (high - low) * 1.1/2
    # L4 = close - 1.5 * (high - low) * 1.1/2
    # H3 = close + 1.25 * (high - low) * 1.1/2
    # L3 = close - 1.25 * (high - low) * 1.1/2
    range_1d = ph - pl
    h4 = pc + 1.5 * range_1d * 1.1 / 2
    l4 = pc - 1.5 * range_1d * 1.1 / 2
    h3 = pc + 1.25 * range_1d * 1.1 / 2
    l3 = pc - 1.25 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for previous day's close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.05 * close[i]  # ATR less than 5% of price
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.3
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 (support break)
            if close[i] < l3_aligned[i] and close[i-1] >= l3_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 (resistance break)
            if close[i] > h3_aligned[i] and close[i-1] <= h3_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above L4 with volume confirmation and volatility filter
            if close[i] > l4_aligned[i] and close[i-1] <= l4_aligned[i-1] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below H4 with volume confirmation and volatility filter
            elif close[i] < h4_aligned[i] and close[i-1] >= h4_aligned[i-1] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals