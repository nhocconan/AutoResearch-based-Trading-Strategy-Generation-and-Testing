#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) with bullish 1w trend and volume spike.
Short when price breaks below lower Donchian(20) with bearish 1w trend and volume spike.
Exit when price returns to middle Donchian line or trend weakens.
Designed for low trade frequency (7-25/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 22:
        return np.zeros(n)
    
    # Calculate 1w EMA21 for trend filter
    close_w = pd.Series(df_weekly['close'].values)
    ema21_w = close_w.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA21 to daily timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_weekly, ema21_w)
    
    # Calculate daily Donchian channels (20-period)
    high_d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_d = (high_d + low_d) / 2.0
    
    # Calculate 20-day average volume for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback
        # Skip if data not ready
        if (np.isnan(high_d[i]) or np.isnan(low_d[i]) or 
            np.isnan(ema21_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with bullish 1w trend and volume spike
            if (close[i] > high_d[i] and 
                close[i] > ema21_aligned[i] and  # Bullish trend: price above weekly EMA21
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with bearish 1w trend and volume spike
            elif (close[i] < low_d[i] and 
                  close[i] < ema21_aligned[i] and  # Bearish trend: price below weekly EMA21
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle Donchian OR trend turns bearish
                if close[i] <= mid_d[i] or close[i] < ema21_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle Donchian OR trend turns bullish
                if close[i] >= mid_d[i] or close[i] > ema21_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%