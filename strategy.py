#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Breakout above/below weekly Camarilla pivot levels (H3/L3) on daily chart with volume confirmation and volatility filter.
# Only take long when price breaks above weekly H3 level, short when breaks below weekly L3 level.
# Exit when price returns to weekly Pivot Point (PP) level.
# Uses volatility filter (ATR < 2.5% of price) and volume confirmation (volume > 1.5x 20-period avg).
# Target: 10-20 trades/year (40-80 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to weekly pivot levels adapting to volatility and volume/vol filters reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
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
    
    # Load weekly data ONCE before loop for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Resistance levels
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    
    # Support levels
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    l3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)  # S3 is L3 in Camarilla
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.025 * close[i]  # ATR less than 2.5% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
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
            # Enter long: price breaks above R3 level with volume confirmation and volatility filter
            if close[i] > r3_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S3 level with volume confirmation and volatility filter
            elif close[i] < l3_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals