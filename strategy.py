#!/usr/bin/env python3
"""
12h_Trix_Volume_Trend
Hypothesis: 12h TRIX momentum filtered by 1w trend and volume spikes captures momentum bursts in both bull and bear markets. TRIX crossing zero indicates acceleration, volume confirms institutional participation, and 1w EMA ensures we trade with the dominant trend. Designed for low trade frequency (~15-30/year) to minimize fee drag while capturing strong moves.
"""

name = "12h_Trix_Volume_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # === 12h TRIX for momentum ===
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.diff() / ema3.shift(1) * 100  # percentage change
    trix_values = trix.fillna(0).values
    
    # === 1w EMA34 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume filter: 2.0x 20 EMA ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers TRIX calculation)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume spike
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                close[i] > ema34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume spike
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_values[i] < 0 and trix_values[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_values[i] > 0 and trix_values[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals