#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian channel breakout with weekly trend filter and volume confirmation.
Long when price breaks above 6h Donchian upper band with bullish weekly trend (price > weekly EMA50) and volume spike.
Short when price breaks below 6h Donchian lower band with bearish weekly trend (price < weekly EMA50) and volume spike.
Exit when price returns to 6h Donchian middle band or trend reverses.
Designed for low trade frequency (12-30/year) to minimize fee drift while capturing breakouts in all regimes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 6h Donchian channel (20-period)
    donch_len = 20
    high_roll = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    low_roll = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    upper = high_roll
    lower = low_roll
    middle = (upper + lower) / 2.0
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donch_len, n):  # Start after Donchian lookback
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above upper band with bullish weekly trend and volume spike
            if (close[i] > upper[i] and 
                close[i] > ema50_1w_aligned[i] and  # Bullish trend: price above weekly EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with bearish weekly trend and volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema50_1w_aligned[i] and  # Bearish trend: price below weekly EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle band OR trend turns bearish
                if close[i] <= middle[i] or close[i] < ema50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle band OR trend turns bullish
                if close[i] >= middle[i] or close[i] > ema50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_WeeklyEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0
#%%