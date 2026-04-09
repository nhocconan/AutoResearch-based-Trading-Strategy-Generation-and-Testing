#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v5
# Hypothesis: Breakout above/below 1-day Camarilla pivot levels (H3/L3) on 4h chart with volume confirmation and volatility filter.
# Only take long when price breaks above H3 level, short when breaks below L3 level.
# Exit when price returns to Pivot Point (PP) level.
# Uses volatility filter (ATR < 3.5% of price) and volume confirmation (volume > 1.3x 20-period avg).
# Reduced trade frequency by tightening volume confirmation to 1.5x and adding ADX(14) > 20 filter to avoid chop.
# Target: 20-35 trades/year (80-140 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to pivot levels adapting to volatility and volume/vol filters reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v5"
timeframe = "4h"
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
    
    # Calculate ADX(14) for trend strength filter
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    tr_ma = np.zeros(n)
    plus_dm_ma = np.zeros(n)
    minus_dm_ma = np.zeros(n)
    # Initial values
    tr_ma[0] = tr[0]
    plus_dm_ma[0] = plus_dm[0]
    minus_dm_ma[0] = minus_dm[0]
    # Smoothing
    for i in range(1, n):
        tr_ma[i] = 0.9 * tr_ma[i-1] + 0.1 * tr[i]
        plus_dm_ma[i] = 0.9 * plus_dm_ma[i-1] + 0.1 * plus_dm[i]
        minus_dm_ma[i] = 0.9 * minus_dm_ma[i-1] + 0.1 * minus_dm[i]
    
    # Avoid division by zero
    dx = np.zeros(n)
    for i in range(n):
        if tr_ma[i] != 0:
            dx[i] = abs(plus_dm_ma[i] - minus_dm_ma[i]) / (tr_ma[i] + 1e-10) * 100
        else:
            dx[i] = 0
    
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = 0.9 * adx[i-1] + 0.1 * dx[i]
    
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
    
    # Resistance levels (H3)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Support levels (L3)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Align 1d levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
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
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.035 * close[i]  # ATR less than 3.5% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average (tightened from 1.3x)
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend strength filter: ADX > 20 to avoid chop
        trend_ok = adx[i] > 20
        
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
            if close[i] > r3_aligned[i] and vol_ok and vol_filter and trend_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S3 level with volume confirmation and volatility filter
            elif close[i] < l3_aligned[i] and vol_ok and vol_filter and trend_ok:
                position = -1
                signals[i] = -0.25
    
    return signals