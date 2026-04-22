#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA20 to 6h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Load daily data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    
    # Upper band: highest high of last 20 days
    upper_d = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower_d = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_d)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above weekly Donchian upper with bullish weekly trend and volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and  # Bullish trend: price above weekly EMA20
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower with bearish weekly trend and volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and  # Bearish trend: price below weekly EMA20
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to the opposite Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly Donchian lower band
                if close[i] <= lower_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly Donchian upper band
                if close[i] >= upper_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "6h"
leverage = 1.0