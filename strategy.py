#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA20 trend filter and volume confirmation
# Long when price breaks above Donchian high(20), 12h EMA20 rising, volume > 1.5x average
# Short when price breaks below Donchian low(20), 12h EMA20 falling, volume > 1.5x average
# Uses 4h for entry timing, 12h for trend filter to avoid whipsaws
# Targets 75-200 total trades over 4 years (19-50/year) for low fee drag

name = "4h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h high/low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema20_12h_val = ema20_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 12h uptrend, volume spike
            if high_val > donchian_high_val and ema20_12h_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 12h downtrend, volume spike
            elif low_val < donchian_low_val and ema20_12h_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or 12h trend down
            if low_val < donchian_low_val or ema20_12h_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or 12h trend up
            if high_val > donchian_high_val or ema20_12h_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals