#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1d volume spike for confirmation
# Long when price breaks above 4h Donchian upper channel AND 1d volume > 2.0x 20-period volume EMA
# Short when price breaks below 4h Donchian lower channel AND 1d volume > 2.0x 20-period volume EMA
# Uses session filter (08-20 UTC) to avoid low-liquidity periods
# Target: 15-30 trades/year by requiring confluence of 4h breakout and 1d volume spike
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends
# Volume spike filter confirms institutional interest reducing false breakouts

name = "1h_Donchian4h_Volume1d_Spike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoids datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF Donchian - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_roll_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = high_roll_4h
    donchian_lower_4h = low_roll_4h
    
    # Align 4h Donchian to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d data for volume spike filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume EMA (20-period) for spike detection
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ema_20_1d * 2.0)  # Volume at least 2.0x average
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 4h Donchian upper AND 1d volume spike
            if close[i] > donchian_upper_aligned[i] and volume_spike_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below 4h Donchian lower AND 1d volume spike
            elif close[i] < donchian_lower_aligned[i] and volume_spike_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below 4h Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above 4h Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals