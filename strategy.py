#!/usr/bin/env python3
"""
4h_Trix_Volume_Spike_1dTrend
Hypothesis: Use TRIX momentum on 4h with volume spike and 1d trend filter. TRIX filters noise and catches sustained momentum. Volume spike confirms institutional interest. 1d trend ensures trades align with higher timeframe bias. Designed to work in both bull and bear by capturing momentum bursts in trending markets.
"""

name = "4h_Trix_Volume_Spike_1dTrend"
timeframe = "4h"
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
    
    # === TRIX on 4h (12-period EMA of EMA of EMA) ===
    # First EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = percentage change in third EMA
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for TRIX (need ~36 bars for 3x EMA(12))
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(ema34_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: positive TRIX + uptrend + volume spike
            if (trix[i] > 0 and 
                close[i] > ema34_4h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: negative TRIX + downtrend + volume spike
            elif (trix[i] < 0 and 
                  close[i] < ema34_4h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX turns negative or trend breaks
            if trix[i] < 0 or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Exit short: TRIX turns positive or trend breaks
            if trix[i] > 0 or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals