#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA50 for higher timeframe trend alignment (stable in both bull/bear, less whipsaw than shorter HTF)
# Donchian(20) from prior 1d session provides clear breakout levels
# Volume confirmation (>1.5x 20 EMA) filters low-participation false breakouts
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-100 total trades over 4 years = 12-25/year for 1d.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
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
    
    # Get 1d data for Donchian calculation and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian(20) from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low) from previous close_1d
    # We need to calculate rolling window on the 1d data
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (completed 1d bar only)
    # Since we're calculating on 1d data, we need to shift by 1 to avoid look-ahead
    # and then align to 1d timeframe
    donchian_upper = np.roll(high_roll, 1)  # shift by 1 to use previous bar's value
    donchian_lower = np.roll(low_roll, 1)   # shift by 1 to use previous bar's value
    # Set first value to NaN since we don't have previous bar
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + uptrend + volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + downtrend + volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend changes OR weak volume
            midpoint = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2.0
            if (close[i] < midpoint or 
                close[i] < ema50_1w_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend changes OR weak volume
            midpoint = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2.0
            if (close[i] > midpoint or 
                close[i] > ema50_1w_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals