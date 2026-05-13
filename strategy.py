#!/usr/bin/env python3
# Hypothesis: 1h Donchian breakout with 4h trend filter and 1d volume confirmation.
# Long when price breaks above 20-bar Donchian high AND 4h EMA50 is rising AND 1d volume > 1.5 * 20-day average volume.
# Short when price breaks below 20-bar Donchian low AND 4h EMA50 is falling AND 1d volume > 1.5 * 20-day average volume.
# Exit on opposite Donchian break or when 4h EMA50 flips direction.
# Uses discrete position sizing (0.20) to limit fee churn. Designed for 1h timeframe with HTF filters to reduce overtrading.
# Target: 80-150 total trades over 4 years (20-37/year) for 1h timeframe.

name = "1h_Donchian20_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_4h_rising = np.gradient(ema_50_4h_aligned) > 0  # True if rising
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_4h_rising[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Donchian high AND 4h EMA50 rising AND volume spike
            if (close[i] > highest_high[i] and 
                ema_50_4h_rising[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: price < Donchian low AND 4h EMA50 falling AND volume spike
            elif (close[i] < lowest_low[i] and 
                  not ema_50_4h_rising[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Donchian low OR 4h EMA50 stops rising
            if (close[i] < lowest_low[i] or not ema_50_4h_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price > Donchian high OR 4h EMA50 starts rising
            if (close[i] > highest_high[i] or ema_50_4h_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals