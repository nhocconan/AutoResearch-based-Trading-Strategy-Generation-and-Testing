#!/usr/bin/env python3
# 4h_1d_donchian_volume_breakout_v3
# Hypothesis: 4-hour Donchian channel breakout with volume confirmation and daily trend filter.
# Long when: price breaks above Donchian(20) high + volume > 1.5x average + price > daily EMA50.
# Short when: price breaks below Donchian(20) low + volume > 1.5x average + price < daily EMA50.
# Exit when price crosses the midline (10-period average of high/low).
# Designed for low frequency (target: 20-40 trades/year) to minimize fee drag.
# Works in bull markets via breakouts and bear via short breakdowns with trend filter.

import numpy as np
import pandas as pd
from mtrand import seed
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation
    vol_avg = np.zeros(n)
    vol_avg[19] = np.mean(volume[:20])
    for i in range(20, n):
        vol_avg[i] = (vol_avg[i-1] * 19 + volume[i]) / 20
    
    # Donchian channels (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donch_high[i] = np.max(high[:i+1]) if i >= 0 else np.nan
            donch_low[i] = np.min(low[:i+1]) if i >= 0 else np.nan
        else:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Midline for exit (10-period average of high/low)
    midline = np.zeros(n)
    for i in range(n):
        if i < 10:
            midline[i] = (np.mean(high[:i+1]) + np.mean(low[:i+1])) / 2 if i >= 0 else np.nan
        else:
            midline[i] = (np.mean(high[i-9:i+1]) + np.mean(low[i-9:i+1])) / 2
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = np.zeros(len(close_1d))
    ema_1d_50[:] = np.nan
    if len(close_1d) >= 50:
        ema_1d_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d_50[i] = close_1d[i] * 0.0377 + ema_1d_50[i-1] * 0.9623
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any values are NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(volume[i]) or np.isnan(vol_avg[i]) or np.isnan(midline[i]) or np.isnan(ema_1d_50_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price crosses below midline
            if close[i] < midline[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above midline
            if close[i] > midline[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: break above Donchian high + volume confirmation + above daily EMA50
            if (close[i] > donch_high[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                close[i] > ema_1d_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: break below Donchian low + volume confirmation + below daily EMA50
            elif (close[i] < donch_low[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  close[i] < ema_1d_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals