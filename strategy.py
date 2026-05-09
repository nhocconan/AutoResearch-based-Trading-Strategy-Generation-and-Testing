#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# Long when: Bull Power > 0, Bear Power < 0, 1d EMA(50) rising, volume spike (>1.5x 20-period average)
# Short when: Bull Power < 0, Bear Power > 0, 1d EMA(50) falling, volume spike
# Exit when: Elder Ray signals reverse OR price crosses 1d EMA(50)
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 15-30 trades/year.
# Elder Ray measures bull/bear power via EMA(13) and works in trending markets; volume confirms strength.

name = "6h_ElderRay_1dEMA_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, 1d EMA rising, volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bull Power < 0, Bear Power > 0, 1d EMA falling, volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Elder Ray reverses OR price crosses below 1d EMA(50)
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                not ema_rising_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Elder Ray reverses OR price crosses above 1d EMA(50)
            if (bull_power[i] >= 0 or bear_power[i] <= 0 or 
                not ema_falling_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: ema_50_1d_aligned needs to be defined before use in the loop
# Let's add it after computing ema_rising_aligned and ema_falling_aligned
# But since we can't modify the loop after defining it, we'll restructure slightly
# Actually, let's fix this by computing ema_50_1d_aligned properly

# Rewriting the function with proper variable definitions: