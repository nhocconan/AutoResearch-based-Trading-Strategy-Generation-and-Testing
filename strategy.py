#24
#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_Trend
Hypothesis: TRIX (triple exponential moving average crossover) identifies momentum shifts in 1w trend. 
Volume spikes confirm institutional participation. Trend filter from 1w EMA20 ensures alignment with long-term momentum.
Works in bull/bear by trading only in direction of 1w trend. Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "12h_TRIX_VolumeSpike_Trend"
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
    
    # Weekly data for TRIX and trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate TRIX: triple EMA of percentage change
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then percentage change
    def ema(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        res = np.full(len(arr), np.nan)
        multiplier = 2 / (period + 1)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (arr[i] - res[i-1]) * multiplier + res[i-1]
        return res
    
    ema1 = ema(close_1w, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix_raw = np.full(len(close_1w), np.nan)
    for i in range(1, len(ema3)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix_raw[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # 1w EMA20 for trend filter
    ema20_1w = ema(close_1w, 20)
    
    # Volume spike: current 1w volume > 2.0x average volume
    vol_mean_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_mean_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_mean_1w[i] = np.mean(volume_1w[i-19:i+1])
    volume_spike = np.full(len(volume_1w), False)
    for i in range(len(volume_1w)):
        if not np.isnan(vol_mean_1w[i]) and volume_1w[i] > 2.0 * vol_mean_1w[i]:
            volume_spike[i] = True
    
    # Align all indicators to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix_raw)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and momentum conditions
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        trix_positive = trix_aligned[i] > 0
        trix_negative = trix_aligned[i] < 0
        vol_spike = volume_spike_aligned[i] > 0.5  # Boolean as float
        
        if position == 0:
            # Long: TRIX turns positive, in uptrend, with volume spike
            if trix_positive and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns negative, in downtrend, with volume spike
            elif trix_negative and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX turns negative or trend turns down
            if not trix_positive or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX turns positive or trend turns up
            if not trix_negative or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals