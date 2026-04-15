#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h Supertrend trend filter and volume confirmation
# Uses Donchian channels for breakout signals, filtered by 12h Supertrend to ensure trades align with higher timeframe trend.
# Volume confirmation avoids false breakouts. Works in bull markets (upward breakouts in uptrend) and bear markets (downward breakouts in downtrend).
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Supertrend (10, 3.0) on 12h
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        # Update bands
        if close_12h[i-1] > upper_band[i-1]:
            upper_band[i] = hl2[i]
        else:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
            
        if close_12h[i-1] < lower_band[i-1]:
            lower_band[i] = hl2[i]
        else:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Determine trend direction
        if close_12h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_12h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Set Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to 6h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(donchian_period, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            continue
        
        # Long entry: price breaks above upper Donchian channel + 12h uptrend + volume confirmation
        if (close[i] > upper_channel[i] and
            direction_aligned[i] == 1 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Donchian channel + 12h downtrend + volume confirmation
        elif (close[i] < lower_channel[i] and
              direction_aligned[i] == -1 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or trend change
        elif position == 1 and (close[i] < lower_channel[i] or direction_aligned[i] == -1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > upper_channel[i] or direction_aligned[i] == 1):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Supertrend_Volume"
timeframe = "6h"
leverage = 1.0