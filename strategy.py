#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1w trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) with bullish 1w trend and volume spike.
Short when price breaks below lower Donchian(20) with bearish 1w trend and volume spike.
Exit when price returns to the opposite Donchian band or trend weakens.
Designed for low trade frequency (12-37/year) to minimize fee drift and capture major trends.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_w = pd.Series(df_weekly['close'].values)
    ema34_w = close_w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema34_w_aligned = align_htf_to_ltf(prices, df_weekly, ema34_w)
    
    # Calculate 12h Donchian channels (20-period)
    # We need to calculate this on 12h data, but we can use the prices directly
    # since prices is already at 12h resolution
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback periods
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema34_w_aligned[i]) or 
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
            # Long: Price breaks above upper Donchian with bullish 1w trend and volume spike
            if (close[i] > upper[i] and 
                close[i] > ema34_w_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with bearish 1w trend and volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema34_w_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to lower Donchian OR trend turns bearish
                if close[i] < lower[i] or close[i] < ema34_w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to upper Donchian OR trend turns bullish
                if close[i] > upper[i] or close[i] > ema34_w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_UpperLower_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%