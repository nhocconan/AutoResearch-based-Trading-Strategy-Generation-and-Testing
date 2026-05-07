#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 12h volume spike.
# Long when price breaks above 4h Donchian upper AND 12h price > EMA50 AND 12h volume spike.
# Short when price breaks below 4h Donchian lower AND 12h price < EMA50 AND 12h volume spike.
# Uses Donchian breakout for momentum, EMA50 for trend alignment, volume spike for confirmation.
# Designed for 25-40 trades/year to avoid fee drag while capturing strong trends.
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
    
    # Load 12h data for EMA50 trend filter and volume spike
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h volume spike: current volume > 2.0 * 20-period EMA
    vol_12h = df_12h['volume'].values
    vol_ema_20_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_12h = np.where(vol_ema_20_12h > 0, vol_12h / vol_ema_20_12h, 1.0) > 2.0
    
    # Align 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian upper, above EMA50, volume spike
            long_condition = (close[i] > donchian_upper[i]) and (close[i] > ema_50_12h_aligned[i]) and vol_spike_12h_aligned[i]
            # Short condition: break below Donchian lower, below EMA50, volume spike
            short_condition = (close[i] < donchian_lower[i]) and (close[i] < ema_50_12h_aligned[i]) and vol_spike_12h_aligned[i]
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian lower or trend reverses (price < EMA50)
            if (close[i] < donchian_lower[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price breaks above Donchian upper or trend reverses (price > EMA50)
            if (close[i] > donchian_upper[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals