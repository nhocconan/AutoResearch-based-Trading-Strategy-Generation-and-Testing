#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 4h Donchian breakout with volume confirmation
# - Uses 12h Supertrend (ATR=10, multiplier=3) to establish trend direction
# - Uses 4h Donchian breakout (20-period) for entry timing
# - Uses 4h volume spike (>2x 20-period average) for entry confirmation
# - Enters long when 12h Supertrend is bullish and price breaks above 4h Donchian upper with volume
# - Enters short when 12h Supertrend is bearish and price breaks below 4h Donchian lower with volume
# - Exits when price crosses the 12h Supertrend line
# - Designed to capture trend moves with proper filtering to avoid whipsaws
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_12hSupertrend_4hDonchian_Volume"
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
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # Calculate True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR using Wilder's smoothing
    atr = np.zeros_like(high_12h)
    atr[atr_period-1] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Align 12h Supertrend to 4h timeframe
    supertrend_4h = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_4h = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 4h Donchian channels (20-period)
    donch_period = 20
    high_4h = high
    low_4h = low
    
    upper_donch = pd.Series(high_4h).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low_4h).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume filter (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(supertrend_4h[i]) or np.isnan(direction_4h[i]) or 
            np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish 12h Supertrend + price breaks above 4h Donchian upper + volume
            if direction_4h[i] == 1 and close[i] > upper_donch[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish 12h Supertrend + price breaks below 4h Donchian lower + volume
            elif direction_4h[i] == -1 and close[i] < lower_donch[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h Supertrend
            if close[i] < supertrend_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h Supertrend
            if close[i] > supertrend_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals