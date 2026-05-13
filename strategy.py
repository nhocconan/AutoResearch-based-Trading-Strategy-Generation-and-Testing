#!/usr/bin/env python3
name = "4H_Donchian_Breakout_VolumeTrend_1D"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on current timeframe
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Get daily data for 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA20 for trend filter
    ema_20_1d = np.zeros_like(close_1d)
    ema_20_1d[:] = np.nan
    alpha = 2 / (20 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_20_1d[i] = close_1d[i]
        elif np.isnan(ema_20_1d[i-1]):
            ema_20_1d[i] = close_1d[i]
        else:
            ema_20_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_20_1d[i-1]
    
    # Align 1d EMA20 to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period) for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above Donchian high + volume spike + 1d uptrend
            if (close[i] > donchian_high[i] and vol_spike and 
                close[i] > ema_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + volume spike + 1d downtrend
            elif (close[i] < donchian_low[i] and vol_spike and 
                  close[i] < ema_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals