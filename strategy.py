#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-period volume MA.
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-period volume MA.
# Exit when price reverts to Donchian(20) midpoint OR volume drops below average.
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian channels provide clear breakout levels, 1w EMA50 filters for higher-timeframe trend, volume confirms participation.
# Designed to work in both bull (breakouts above EMA50) and bear (breakdowns below EMA50) markets.

name = "12h_Donchian20_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1w EMA50 AND volume spike AND session
            if close[i] > highest_high_20[i] and close[i] > ema_50_1w_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA50 AND volume spike AND session
            elif close[i] < lowest_low_20[i] and close[i] < ema_50_1w_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid[i] or volume[i] < volume_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals