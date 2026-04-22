#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian(40) and ATR(20) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 40:
        return np.zeros(n)
    
    # Calculate Donchian(40) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_40 = pd.Series(high_daily).rolling(window=40, min_periods=40).max().values
    lower_40 = pd.Series(low_daily).rolling(window=40, min_periods=40).min().values
    
    # Calculate ATR(20) from daily data
    close_daily = df_daily['close'].values
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian channels and ATR to 12h timeframe
    upper_40_aligned = align_htf_to_ltf(prices, df_daily, upper_40)
    lower_40_aligned = align_htf_to_ltf(prices, df_daily, lower_40)
    atr_20_aligned = align_htf_to_ltf(prices, df_daily, atr_20)
    
    # Calculate 12h volume average (40-period)
    vol_avg_40 = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_40_aligned[i]) or np.isnan(lower_40_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(vol_avg_40[i])):
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
            # Long: Price breaks above upper Donchian(40) with volume and ATR filter
            if (close[i] > upper_40_aligned[i] and 
                volume[i] > 1.8 * vol_avg_40[i] and
                atr_20_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(40) with volume and ATR filter
            elif (close[i] < lower_40_aligned[i] and 
                  volume[i] > 1.8 * vol_avg_40[i] and
                  atr_20_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
            if position == 1:
                if close[i] < lower_40_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_40_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian40_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0