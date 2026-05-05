#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1h Supertrend for trend direction, 4h Donchian(20) breakout for entry, and volume confirmation.
# Long when price breaks above Donchian upper band AND Supertrend is bullish AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian lower band AND Supertrend is bearish AND volume > 1.5 * avg_volume(20)
# Exit when price touches Donchian midpoint (average of upper and lower bands)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Supertrend on 1h provides timely trend direction with less lag than higher timeframes
# Donchian breakouts capture strong momentum moves
# Volume confirmation validates breakout strength and reduces false signals

name = "4h_1hSupertrend_Donchian20_Volume"
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
    
    # Get 1h data ONCE before loop for Supertrend calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 10:  # Need sufficient data for ATR and Supertrend
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate ATR for Supertrend (period=10)
    tr1 = np.maximum(high_1h[1:] - low_1h[1:], np.abs(high_1h[1:] - close_1h[:-1]))
    tr2 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.max([high_1h[0] - low_1h[0], np.abs(high_1h[0] - close_1h[0]), np.abs(low_1h[0] - close_1h[0])])], np.maximum(tr1, tr2)])
    atr_period = 10
    atr_1h = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Supertrend
    factor = 3.0
    hl2 = (high_1h + low_1h) / 2
    upper_band = hl2 + (factor * atr_1h)
    lower_band = hl2 - (factor * atr_1h)
    
    supertrend = np.zeros_like(close_1h)
    direction = np.ones_like(close_1h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, len(close_1h)):
        if close_1h[i-1] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
        
        if close_1h[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1
    
    # Align 1h Supertrend direction to 4h timeframe (wait for completed 1h bar)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1h, direction)
    
    # Calculate Donchian channels on 4h (period=20)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, Supertrend bullish (direction=1), volume confirmation, in session
            if (close[i] > highest_high[i] and 
                supertrend_dir_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, Supertrend bearish (direction=-1), volume confirmation, in session
            elif (close[i] < lowest_low[i] and 
                  supertrend_dir_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals