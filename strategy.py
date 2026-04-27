#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period) for trend context
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # 1d ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.append([np.nan], high_1d[:-1]))
    tr3 = np.abs(low_1d - np.append([np.nan], low_1d[:-1]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 00-23 UTC (all hours for 12h timeframe)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 0) & (hour <= 23)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high + volume + low volatility
            if (close[i] > donch_high_20_aligned[i] and 
                volume_filter[i] and 
                atr14_1d_aligned[i] < np.nanmedian(atr14_1d_aligned[:i+1]) and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low + volume + low volatility
            elif (close[i] < donch_low_20_aligned[i] and 
                  volume_filter[i] and 
                  atr14_1d_aligned[i] < np.nanmedian(atr14_1d_aligned[:i+1]) and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 1d Donchian low (trend change)
            if close[i] < donch_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 1d Donchian high (trend change)
            if close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Vol_LowVol_Filter"
timeframe = "12h"
leverage = 1.0