#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above upper Donchian band with bullish 12h trend and volume spike.
Short when price breaks below lower Donchian band with bearish 12h trend and volume spike.
Exit when price returns to Donchian middle band or trend weakens.
Designed for low trade frequency (15-25/year) to minimize fee drag and improve generalization.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate 12h EMA25 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema25_12h = close_12h.ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Align EMA25 to 4h timeframe
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2.0
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback periods
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(middle[i]) or np.isnan(ema25_12h_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with bullish 12h trend and volume spike
            if (close[i] > upper[i] and 
                close[i] > ema25_12h_aligned[i] and  # Bullish trend: price above EMA25
                volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with bearish 12h trend and volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema25_12h_aligned[i] and  # Bearish trend: price below EMA25
                  volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle OR trend turns bearish
                if close[i] <= middle[i] or close[i] < ema25_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle OR trend turns bullish
                if close[i] >= middle[i] or close[i] > ema25_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_UpperLower_12hEMA25_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%