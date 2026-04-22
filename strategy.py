#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian(15) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 15:
        return np.zeros(n)
    
    # Calculate Donchian(15) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_15 = pd.Series(high_daily).rolling(window=15, min_periods=15).max().values
    lower_15 = pd.Series(low_daily).rolling(window=15, min_periods=15).min().values
    
    # Align Donchian channels to 12h timeframe
    upper_15_aligned = align_htf_to_ltf(prices, df_daily, upper_15)
    lower_15_aligned = align_htf_to_ltf(prices, df_daily, lower_15)
    
    # Calculate 12h volume average (15-period)
    vol_avg_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_15_aligned[i]) or np.isnan(lower_15_aligned[i]) or 
            np.isnan(vol_avg_15[i])):
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
            # Long: Price breaks above upper Donchian(15) with volume
            if (close[i] > upper_15_aligned[i] and 
                volume[i] > 1.4 * vol_avg_15[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(15) with volume
            elif (close[i] < lower_15_aligned[i] and 
                  volume[i] > 1.4 * vol_avg_15[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
            if position == 1:
                if close[i] < lower_15_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_15_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian15_Volume_Session"
timeframe = "12h"
leverage = 1.0