#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout + 1d EMA trend filter + volume spike
# Long when: price breaks above 4h Donchian(20) upper AND close > 1d EMA50 AND volume > 2x 20-bar MA
# Short when: price breaks below 4h Donchian(20) lower AND close < 1d EMA50 AND volume > 2x 20-bar MA
# Exit when: price touches opposite Donchian band OR volume drops below average
# Uses Donchian for structure, 1d EMA for trend filter, volume for confirmation
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_Donchian_1dEMA_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channel (20-period) ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower on 4h
    donch_hi = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for completed 4h bar)
    donch_hi_aligned = align_htf_to_ltf(prices, df_4h, donch_hi)
    donch_lo_aligned = align_htf_to_ltf(prices, df_4h, donch_lo)
    
    # Get 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
        volume_normal = volume < vol_ma_20  # for exit condition
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_normal = np.ones(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donch_hi_aligned[i]) or np.isnan(donch_lo_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper + above 1d EMA50 + volume spike
            if (close[i] > donch_hi_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below Donchian lower + below 1d EMA50 + volume spike
            elif (close[i] < donch_lo_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian lower OR volume normalizes
            if (close[i] < donch_lo_aligned[i] or volume_normal[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price touches Donchian upper OR volume normalizes
            if (close[i] > donch_hi_aligned[i] or volume_normal[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals