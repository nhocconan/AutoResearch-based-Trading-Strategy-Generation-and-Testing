#!/usr/bin/env python3
# 12h_1d_ema_breakout_v1
# Hypothesis: Breakout above/below 1d EMA(200) on 12h chart with volume confirmation and ATR filter.
# Long when price closes above EMA(200) with volume > 1.5x average, short when price closes below EMA(200) with volume > 1.5x average.
# Exit when price crosses back over EMA(200).
# Uses volume spike to confirm institutional interest and avoid false breakouts.
# Designed for low trade frequency (<30/year) to minimize fee drag in ranging/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need 200 for EMA + 10 buffer
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
    
    # Load 1d data ONCE before loop for EMA(200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA(200)
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (200 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA(200) to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    
    for i in range(210, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.05 * close[i]  # ATR less than 5% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price closes below EMA(200)
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA(200)
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above EMA(200) with volume confirmation and volatility filter
            if close[i] > ema_1d_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below EMA(200) with volume confirmation and volatility filter
            elif close[i] < ema_1d_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals