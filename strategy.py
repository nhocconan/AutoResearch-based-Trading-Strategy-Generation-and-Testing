#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w Supertrend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band AND 1w Supertrend is bullish AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian lower band AND 1w Supertrend is bearish AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides strong structural breakout levels that reduce whipsaw
# 1w Supertrend filter ensures we trade with the dominant weekly trend (works in both bull and bear)
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading

name = "12h_1dDonchian20_1wSupertrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().values
    donchian_mid_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Align 1d Donchian to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    # Get 1w data ONCE before loop for Supertrend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for ATR
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(10) for Supertrend
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w Supertrend with parameters (10, 3.0)
    factor = 3.0
    hl2 = (high_1w + low_1w) / 2
    upperband = hl2 + (factor * atr_1w)
    lowerband = hl2 - (factor * atr_1w)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    # Calculate Supertrend
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1w[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upperband[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lowerband[i]
        elif direction[i] == 1:
            supertrend[i] = max(upperband[i], supertrend[i-1])
        else:
            supertrend[i] = min(lowerband[i], supertrend[i-1])
    
    # Align 1w Supertrend direction to 12h timeframe (wait for completed 1w bar)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(supertrend_direction_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band, 1w Supertrend bullish (direction=1), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                supertrend_direction_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band, 1w Supertrend bearish (direction=-1), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  supertrend_direction_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals