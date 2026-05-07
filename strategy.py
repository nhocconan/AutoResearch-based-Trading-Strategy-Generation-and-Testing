#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend + 1d volume spike filter.
# Long when price breaks above 4h Donchian upper channel AND 12h EMA(50) rising AND 1d volume spike.
# Short when price breaks below 4h Donchian lower channel AND 12h EMA(50) falling AND 1d volume spike.
# Uses volume spike for momentum confirmation and EMA for trend filter.
# Designed for fewer trades (target: 20-30/year) to reduce fee drag and improve generalization.
# Works in both bull and bear markets by following 4h price action with trend and volume filters.
name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Load 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50)
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Donchian channels
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_period)  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above upper channel, EMA rising, volume spike
            ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1]
            long_condition = (close[i] > upper_channel[i]) and ema_rising and vol_spike_1d_aligned[i]
            # Short condition: break below lower channel, EMA falling, volume spike
            ema_falling = ema_12h_aligned[i] < ema_12h_aligned[i-1]
            short_condition = (close[i] < lower_channel[i]) and ema_falling and vol_spike_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below lower channel or EMA turns down
            if (close[i] < lower_channel[i]) or (ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above upper channel or EMA turns up
            if (close[i] > upper_channel[i]) or (ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals