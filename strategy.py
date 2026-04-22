#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian(10) and ATR(10) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate Donchian(10) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_10 = pd.Series(high_daily).rolling(window=10, min_periods=10).max().values
    lower_10 = pd.Series(low_daily).rolling(window=10, min_periods=10).min().values
    
    # Calculate ATR(10) from daily data
    close_daily = df_daily['close'].values
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align Donchian channels and ATR to 12h timeframe
    upper_10_aligned = align_htf_to_ltf(prices, df_daily, upper_10)
    lower_10_aligned = align_htf_to_ltf(prices, df_daily, lower_10)
    atr_10_aligned = align_htf_to_ltf(prices, df_daily, atr_10)
    
    # Calculate 12h volume average (10-period)
    vol_avg_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_10_aligned[i]) or np.isnan(lower_10_aligned[i]) or 
            np.isnan(atr_10_aligned[i]) or np.isnan(vol_avg_10[i])):
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
            # Long: Price breaks above upper Donchian(10) with volume and ATR filter
            if (close[i] > upper_10_aligned[i] and 
                volume[i] > 1.5 * vol_avg_10[i] and
                atr_10_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(10) with volume and ATR filter
            elif (close[i] < lower_10_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_10[i] and
                  atr_10_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
            if position == 1:
                if close[i] < lower_10_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_10_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian10_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0