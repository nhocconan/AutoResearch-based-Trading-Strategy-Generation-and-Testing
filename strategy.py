#!/usr/bin/env python3
# 6h_TRIX_ZeroCross_1dTrend_VolumeSpike
# Hypothesis: TRIX zero-cross with 1d EMA trend filter and volume spike confirmation.
# TRIX (triple smoothed EMA) filters out noise and identifies momentum shifts.
# Zero-cross signals trend changes; 1d EMA filter ensures alignment with higher timeframe trend.
# Volume spike confirms institutional participation. Designed for 6h timeframe to balance
# signal frequency and noise reduction, targeting 12-35 trades/year.
# Works in bull/bear markets: trend filter avoids counter-trend trades, volume adds confirmation.

name = "6h_TRIX_ZeroCross_1dTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Calculate TRIX (15-period triple EMA) - momentum oscillator
    def calculate_trix(price_array, period=15):
        if len(price_array) < period * 3:
            return np.full_like(price_array, np.nan)
        # First EMA
        ema1 = np.full_like(price_array, np.nan)
        alpha = 2 / (period + 1)
        ema1[period-1] = np.mean(price_array[0:period])
        for i in range(period, len(price_array)):
            ema1[i] = alpha * price_array[i] + (1 - alpha) * ema1[i-1]
        # Second EMA
        ema2 = np.full_like(price_array, np.nan)
        ema2[2*period-1] = np.mean(ema1[period-1:2*period])
        for i in range(2*period, len(price_array)):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        # Third EMA
        ema3 = np.full_like(price_array, np.nan)
        ema3[3*period-1] = np.mean(ema2[2*period-1:3*period])
        for i in range(3*period, len(price_array)):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
        # TRIX = % change of third EMA
        trix = np.full_like(price_array, np.nan)
        for i in range(3*period, len(price_array)):
            if ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
        return trix
    
    trix = calculate_trix(close, 15)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        alpha = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)  # Ensure TRIX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or  # Need previous TRIX for zero-cross
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (price > EMA34) AND volume spike
            if (trix[i-1] <= 0 and trix[i] > 0 and  # Zero-cross up
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (price < EMA34) AND volume spike
            elif (trix[i-1] >= 0 and trix[i] < 0 and  # Zero-cross down
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (price < EMA34)
            if (trix[i-1] >= 0 and trix[i] < 0) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (price > EMA34)
            if (trix[i-1] <= 0 and trix[i] > 0) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals