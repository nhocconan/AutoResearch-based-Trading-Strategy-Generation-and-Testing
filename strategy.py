#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + session filter
# Long when price breaks above 4h Donchian upper (20) AND volume > 1.5 * 20-bar avg AND hour in [8,20) UTC
# Short when price breaks below 4h Donchian lower (20) AND volume > 1.5 * 20-bar avg AND hour in [8,20) UTC
# Exit when price crosses 4h Donchian middle (10-period avg of upper/lower) OR volume drops below avg
# Uses discrete sizing 0.20 to minimize fee churn. 4h provides trend filter, 1h for precise entry timing.
# Session filter reduces noise trades during low-volume hours. Target: 60-150 total trades over 4 years.

name = "1h_DonchianBreakout_4hVolume_SessionFilter_v1"
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
    open_time = prices['open_time'].values  # already datetime64[ms]
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, 20)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, 20)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian middle = average of upper and lower
    donchian_middle = (donchian_high + donchian_low) / 2.0
    
    # Align HTF indicators to 1h timeframe (wait for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour < 20)  # UTC 8-20
        
        if position == 0:
            # Entry conditions: Donchian breakout + volume spike + session
            # Long: price breaks above Donchian upper
            if close[i] > donchian_high_aligned[i] and volume_spike[i] and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower
            elif close[i] < donchian_low_aligned[i] and volume_spike[i] and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR volume drops below average
            if close[i] < donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR volume drops below average
            if close[i] > donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals