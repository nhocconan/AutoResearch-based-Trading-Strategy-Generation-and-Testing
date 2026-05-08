#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high and 12h EMA50 is rising
# Short when price breaks below Donchian(20) low and 12h EMA50 is falling
# Volume confirmation: current volume > 1.5x 20-period average
# Exit when price crosses Donchian midline or trend reverses
# Targets 80-150 total trades over 4 years (20-38/year) for optimal fee drag

name = "4h_Donchian20_12hEMA50_Volume"
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
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_roll = (high_roll + low_roll) / 2
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_prev = np.roll(ema50_12h, 1)
    ema50_12h_prev[0] = ema50_12h[0]
    ema50_12h_rising = ema50_12h > ema50_12h_prev
    ema50_12h_falling = ema50_12h < ema50_12h_prev
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_val = high_roll[i]
        low_val = low_roll[i]
        mid_val = mid_roll[i]
        ema50_12h_val = ema50_12h[i]
        ema50_rising = ema50_12h_rising[i]
        ema50_falling = ema50_12h_falling[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 12h EMA rising, volume confirmation
            if close_val > high_val and ema50_rising and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 12h EMA falling, volume confirmation
            elif close_val < low_val and ema50_falling and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midline or 12h EMA starts falling
            if close_val < mid_val or ema50_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midline or 12h EMA starts rising
            if close_val > mid_val or ema50_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals