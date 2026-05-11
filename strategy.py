# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Pivot_Reversal_Backtest
Hypothesis: Fade intraday reversals from previous day's open with 1d EMA trend filter.
In ranging markets (common in 2025), price tends to revert to the day's open after strong moves.
We use the 1d EMA to determine trend direction: only fade when price deviates significantly
from the open in the opposite trend direction. Volume spike confirms exhaustion.
Works in both bull and bear markets by fading extremes within the prevailing trend.
Target: 15-30 trades/year per symbol.
"""

name = "6h_Pivot_Reversal_Backtest"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for Trend Filter and Open Reference ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA for trend filter (20-period)
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Previous day's open as mean reversion target
    open_1d = df_1d['open'].values
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    
    # === 60-period ATR for deviation threshold (on 6h data) ===
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr60 = pd.Series(tr).rolling(window=60, min_periods=60).mean().values
    
    # === Volume Spike Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA and ATR)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(open_1d_aligned[i]) or 
            np.isnan(atr60[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from day's open in ATR units
        deviation = (close[i] - open_1d_aligned[i]) / atr60[i]
        
        if position == 0:
            # Fade down: price is significantly above open AND trend is down (price < EMA)
            if deviation > 1.5 and close[i] < ema20_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            # Fade up: price is significantly below open AND trend is up (price > EMA)
            elif deviation < -1.5 and close[i] > ema20_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Long exit: price returns to near open or trend breaks down
            if deviation > -0.5 or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price returns to near open or trend breaks up
            if deviation < 0.5 or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals