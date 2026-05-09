#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Uses actual Donchian channel (20-day high/low) to capture long-term breakouts.
# Long when price breaks above 20-day high with 1w uptrend and volume spike.
# Short when price breaks below 20-day low with 1w downtrend and volume spike.
# Designed to work in both bull (follow 1w uptrend) and bear (follow 1w downtrend) markets.
# Target: 15-25 trades/year to minimize fee drag on 1d timeframe.
name = "1d_Donchian20_1wEMA_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 20-day Donchian and 1w EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels
    # Highest high and lowest low over past 20 days
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 2.0x 20-day EMA (high threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above 20-day high + 1w uptrend + volume spike
            if (price > high_20[i] and price > ema_20_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low + 1w downtrend + volume spike
            elif (price < low_20[i] and price < ema_20_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 20-day high or trend reverses
            if price < high_20[i] or price < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 20-day low or trend reverses
            if price > low_20[i] or price > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals