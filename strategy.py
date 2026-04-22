#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian breakout (20) with weekly trend filter (EMA21) and volume confirmation.
Long when price breaks above Donchian upper with bullish weekly trend and volume spike.
Short when price breaks below Donchian lower with bearish weekly trend and volume spike.
Exit when price returns to Donchian middle (midline).
Designed for low trade frequency (<15/year) to minimize fee drag in bear markets.
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
    
    # Load 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower (20-period)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1d, ema21_1w)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with bullish weekly trend and volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema21_1w_aligned[i] and  # Bullish trend: price above weekly EMA21
                volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with bearish weekly trend and volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema21_1w_aligned[i] and  # Bearish trend: price below weekly EMA21
                  volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian middle
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle
                if close[i] <= donchian_middle_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle
                if close[i] >= donchian_middle_aligned[i]:
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