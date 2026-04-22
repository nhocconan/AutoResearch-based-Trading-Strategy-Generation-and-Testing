#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian and ATR - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Donchian(20) channels
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility filter
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian and ATR to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    atr_aligned = align_htf_to_ltf(prices, df_daily, atr)
    
    # Load 1h data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    vol_1h = df_1h['volume'].values
    vol_avg_20_1h = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_avg_20_1h)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_avg_20_1h_aligned[i])):
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
            # Long: Price breaks above upper Donchian(20) with volume > 2x average and ATR > 0
            if (close[i] > upper_20_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20_1h_aligned[i] and
                atr_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) with volume > 2x average and ATR > 0
            elif (close[i] < lower_20_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20_1h_aligned[i] and
                  atr_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel or ATR-based stop
            if position == 1:
                if close[i] < lower_20_aligned[i] or close[i] < (upper_20_aligned[i] - 1.5 * atr_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i] or close[i] > (lower_20_aligned[i] + 1.5 * atr_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_VolumeATR_Filter_Session"
timeframe = "4h"
leverage = 1.0