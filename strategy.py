#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 12h EMA50 is rising AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) low AND 12h EMA50 is falling AND volume > 1.5x 20-period MA.
# Exit when price touches Donchian(20) midpoint OR trend reverses (EMA50 slope changes sign).
# Uses 4h timeframe to achieve 75-200 total trades over 4 years with strict entry conditions.
# Donchian channels provide structure, EMA50 filters for trending markets, volume confirms participation.
# Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets.

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA50 slope (rising/falling)
    ema_slope_12h = np.diff(ema_50_12h, prepend=ema_50_12h[0])
    ema_rising = ema_slope_12h > 0
    ema_falling = ema_slope_12h < 0
    
    # Align 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    
    # Calculate 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 4h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA50 rising AND volume spike AND session
            if close[i] > highest_20[i] and ema_rising_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA50 falling AND volume spike AND session
            elif close[i] < lowest_20[i] and ema_falling_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian midpoint OR EMA50 stops rising
            if close[i] <= midpoint_20[i] or not ema_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian midpoint OR EMA50 stops falling
            if close[i] >= midpoint_20[i] or not ema_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals