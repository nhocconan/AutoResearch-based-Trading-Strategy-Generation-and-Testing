#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# Long when: Price breaks above Donchian upper (20) AND 12h EMA(50) > 12h EMA(100) (uptrend) AND volume > 1.5x 20-period average volume
# Short when: Price breaks below Donchian lower (20) AND 12h EMA(50) < 12h EMA(100) (downtrend) AND volume > 1.5x 20-period average volume
# Exit when price returns to Donchian middle (mean of upper/lower) or opposite breakout occurs
# Donchian breakout captures sustained momentum after consolidation
# 12h EMA filter ensures we trade with the higher timeframe trend
# Volume confirmation filters weak breakouts
# Works in both bull and bear markets by aligning with 12h trend direction
# Target: 100-180 total trades over 4 years (25-45/year) with discrete sizing 0.25

name = "4h_DonchianBreakout_12hEMATrend_Volume"
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
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:  # Need enough for EMA(100)
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_100_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    # Calculate Donchian channels (20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_donchian = highest_20
    lower_donchian = lowest_20
    middle_donchian = (upper_donchian + lower_donchian) / 2
    
    # Calculate 20-period average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_100_aligned[i]) or 
            np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(avg_vol_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        if position == 0:
            # Long: Break above upper Donchian in uptrend with volume confirmation
            if close[i] > upper_donchian[i] and ema_50_aligned[i] > ema_100_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian in downtrend with volume confirmation
            elif close[i] < lower_donchian[i] and ema_50_aligned[i] < ema_100_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian or reverse breakout
            if close[i] < middle_donchian[i] or close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian or reverse breakout
            if close[i] > middle_donchian[i] or close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals