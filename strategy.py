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
    
    # Load daily data for Donchian(20) and ATR(14) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(high_daily) - pd.Series(low_daily)
    tr2 = abs(pd.Series(high_daily) - pd.Series(pd.Series(close_daily).shift(1)))
    tr3 = abs(pd.Series(low_daily) - pd.Series(pd.Series(close_daily).shift(1)))
    close_daily = df_daily['close'].values
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian channels and ATR to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_daily, atr_14)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above upper Donchian(20) with volume and volatility filter
            if (close[i] > upper_20_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                atr_14_aligned[i] > 0):  # ATR > 0 ensures volatility present
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) with volume and volatility filter
            elif (close[i] < lower_20_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  atr_14_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
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

name = "6H_Donchian20_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0