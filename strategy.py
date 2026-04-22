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
    
    # Load daily data for Donchian(20) and Camarilla pivots - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily Camarilla pivot levels
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    
    # Camarilla levels
    r4 = daily_close + range_hl * 1.1 / 2
    r3 = daily_close + range_hl * 1.1 / 4
    s3 = daily_close - range_hl * 1.1 / 4
    s4 = daily_close - range_hl * 1.1 / 2
    
    # Align indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
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
            # Long: Price breaks above upper Donchian(20) AND above R3 with volume
            if (close[i] > upper_20_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) AND below S3 with volume
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian channel
            if position == 1:
                if close[i] < lower_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Donchian20_Camarilla_R3S3_Volume_Session"
timeframe = "6h"
leverage = 1.0