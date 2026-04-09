#!/usr/bin/env python3
# 6h_1d_rsi_volume_reversal_v1
# Hypothesis: 6-hour RSI mean reversion with volume confirmation and daily trend filter.
# Long when RSI(14) < 30, price > daily EMA50, and volume > 1.5x 20-bar average.
# Short when RSI(14) > 70, price < daily EMA50, and volume > 1.5x 20-bar average.
# Exit when RSI returns to neutral zone (40-60).
# Works in bull markets via pullbacks to EMA50 and in bear markets via bounces from oversold/overbought.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_volume_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = close_1d[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = ema
        for i in range(50, len(close_1d)):
            ema = (close_1d[i] - ema) * multiplier + ema
            ema_50_1d[i] = ema
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h close
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Wilder's smoothing
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
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
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral zone (>= 40)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral zone (<= 60)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI oversold with trend and volume filters
            if (rsi[i] < 30 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: RSI overbought with trend and volume filters
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals