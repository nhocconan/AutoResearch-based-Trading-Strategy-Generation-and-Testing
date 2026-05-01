#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian(20) breakout + 1d volume spike + session filter (08-20 UTC)
# Uses 4h Donchian channels for trend structure and breakout signals
# Volume spike on 1d confirms institutional participation
# Session filter reduces noise during low-liquidity hours
# Designed for 1h timeframe with tight entry conditions to limit trades to 60-150 over 4 years
# Works in both bull and bear markets by trading breakouts in direction of 4h trend

name = "1h_Donchian20_1dVolume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    
    # 4h HTF data for Donchian channels (trend structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d HTF data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper channel = highest high over 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower channel = lowest low over 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Donchian (20 periods)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above 4h Donchian upper channel with volume spike
            if close[i] > donchian_high_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below 4h Donchian lower channel with volume spike
            elif close[i] < donchian_low_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to midpoint of 4h Donchian channel
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to midpoint of 4h Donchian channel
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals