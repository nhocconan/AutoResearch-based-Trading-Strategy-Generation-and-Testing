#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 6h ATR-based Donchian breakout for entry
# Long when: 12h Supertrend is bullish AND price breaks above 6h Donchian(20) upper band AND 6h volume > 1.3 * avg_volume(20)
# Short when: 12h Supertrend is bearish AND price breaks below 6h Donchian(20) lower band AND 6h volume > 1.3 * avg_volume(20)
# Exit when: price crosses the 6h Donchian(20) midpoint OR opposite Supertrend signal occurs
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Supertrend provides strong trend filter reducing false breakouts
# Donchian breakout captures momentum in trending markets
# Volume confirmation ensures breakout conviction
# Works in bull markets (continuation breakouts) and bear markets (continuation breakdowns)

name = "6h_12hSupertrend_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:  # Need sufficient data for ATR and Supertrend
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for Supertrend
    atr_period = 10
    tr1 = pd.Series(high_12h).rolling(2).max() - pd.Series(low_12h).rolling(2).min()
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 12h Supertrend
    factor = 3.0
    hl2_12h = (high_12h + low_12h) / 2.0
    upperband_12h = hl2_12h + (factor * atr_12h)
    lowerband_12h = hl2_12h - (factor * atr_12h)
    
    supertrend_12h = np.zeros_like(close_12h)
    direction_12h = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend_12h[0] = upperband_12h[0]
    direction_12h[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend_12h[i-1]:
            supertrend_12h[i] = max(upperband_12h[i], supertrend_12h[i-1])
            direction_12h[i] = 1
        else:
            supertrend_12h[i] = min(lowerband_12h[i], supertrend_12h[i-1])
            direction_12h[i] = -1
    
    # Align 12h Supertrend direction to 6h timeframe (wait for completed 12h bar)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate 6h Donchian(20) channels
    donchian_period = 20
    upperband_6h = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowerband_6h = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    midpoint_6h = (upperband_6h + lowerband_6h) / 2.0
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(direction_12h_aligned[i]) or np.isnan(upperband_6h[i]) or 
            np.isnan(lowerband_6h[i]) or np.isnan(midpoint_6h[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h Supertrend bullish, price breaks above 6h Donchian upper band, volume spike
            if (direction_12h_aligned[i] == 1 and 
                close[i] > upperband_6h[i] and close[i-1] <= upperband_6h[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h Supertrend bearish, price breaks below 6h Donchian lower band, volume spike
            elif (direction_12h_aligned[i] == -1 and 
                  close[i] < lowerband_6h[i] and close[i-1] >= lowerband_6h[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 6h Donchian midpoint OR Supertrend turns bearish
            if close[i] < midpoint_6h[i] or direction_12h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 6h Donchian midpoint OR Supertrend turns bullish
            if close[i] > midpoint_6h[i] or direction_12h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals