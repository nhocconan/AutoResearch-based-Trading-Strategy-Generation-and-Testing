#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Hypothesis: Breakout above/below daily Camarilla pivot levels with 1d trend filter (EMA 200) on 12h timeframe.
# Only take long when price > 1d EMA(200), short when price < 1d EMA(200).
# Enter long when price breaks above daily Camarilla H3 level, enter short when price breaks below daily L3 level.
# Exit when price crosses 1d EMA(200) in opposite direction.
# Uses volume confirmation (volume > 1.3x 20-period average) to filter breakouts.
# Target: 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to trend filter + volume filter reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
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
    
    # Calculate ATR(14) for volatility filter (not used in entry but for risk)
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
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = C + (H-L)*1.1/2, L3 = C - (H-L)*1.1/2
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        h_l = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + h_l * 1.1 / 2
        camarilla_l3[i] = close_1d[i] - h_l * 1.1 / 2
    
    # Calculate 1d EMA(200) for trend filter
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (200 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d Camarilla levels and EMA to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.3
        
        # Trend filter: price > 1d EMA for longs, price < 1d EMA for shorts
        trend_long = close[i] > ema_1d_aligned[i]
        trend_short = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily Camarilla H3 with volume confirmation and trend filter
            if close[i] > camarilla_h3_aligned[i] and vol_ok and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily Camarilla L3 with volume confirmation and trend filter
            elif close[i] < camarilla_l3_aligned[i] and vol_ok and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals