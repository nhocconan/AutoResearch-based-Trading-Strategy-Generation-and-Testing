#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1w EMA34 for higher timeframe trend alignment (stable in both bull/bear)
# Donchian(20) from prior 1d session provides clear breakout levels
# Volume confirmation (>2.0x 20 EMA) filters low-participation false breakouts
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 30-100 total trades over 4 years = 7-25/year for 1d.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "1d_Donchian20_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (completed 1w bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian(20) levels from previous 1d bar
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    
    # Rolling max/min of high/low over 20 periods
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if close[i] > donchian_high[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend changes OR weak volume
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            
            if (close[i] < midpoint or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend changes OR weak volume
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            
            if (close[i] > midpoint or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals