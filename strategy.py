#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) with bullish 1d trend and volume spike.
Short when Williams %R > -20 (overbought) with bearish 1d trend and volume spike.
Exit when Williams %R returns to -50 (mean reversion).
Uses 1d EMA50 for trend filter to avoid counter-trend trades.
Designed for low trade frequency (15-30/year) to minimize fee drag in ranging markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R calculation and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max().values
    lowest_low = low_1d.rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R lookback
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_aligned[i]) or 
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
            # Long: Williams %R < -80 (oversold) with bullish 1d trend and volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with bearish 1d trend and volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to -50 (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to -50 from below
                if williams_r_aligned[i] >= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to -50 from above
                if williams_r_aligned[i] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extreme_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%