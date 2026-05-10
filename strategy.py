#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_1dTrend
Hypothesis: TRIX (12) crossing above/below zero with volume spikes and 1d EMA34 trend filter.
TRIX filters noise and identifies momentum shifts. Volume spikes confirm breakout strength.
In trending markets (1d EMA34), we take TRIX signals in trend direction.
Works in bull (TRIX up in uptrend) and bear (TRIX down in downtrend).
Target: 20-50 trades/year to minimize fee drag.
"""

name = "4h_TRIX_VolumeSpike_1dTrend"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume spike detection
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # TRIX calculation (12-period)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then % change
    ema1 = np.full(len(close), np.nan)
    ema2 = np.full(len(close), np.nan)
    ema3 = np.full(len(close), np.nan)
    trix = np.full(len(close), np.nan)
    
    if len(close) >= 12:
        # First EMA
        ema1[11] = np.mean(close[:12])
        alpha = 2 / (12 + 1)
        for i in range(12, len(close)):
            ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]
        
        # Second EMA of EMA1
        ema2[23] = np.mean(ema1[12:24])  # Start at index 23 (24th element)
        for i in range(24, len(close)):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        
        # Third EMA of EMA2
        ema3[35] = np.mean(ema2[24:36])  # Start at index 35 (36th element)
        for i in range(36, len(close)):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
        
        # TRIX = % change of ema3
        for i in range(36, len(close)):
            if ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 1)  # Need TRIX and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h periods in 1d
        volume_spike = volume[i] > 2.0 * vol_4h_approx
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in uptrend
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike in downtrend
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend reversal
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend reversal
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals