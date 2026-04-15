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
    
    # 1d data for 20-period Donchian channel and 14-period ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channel (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    
    # Calculate 1d ATR (14-period)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume median (30-period)
    vol_median_1d = pd.Series(volume_1d).rolling(window=30, min_periods=30).median()
    
    # Align all 1d indicators to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # 12h data for trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_median_1d_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            continue
        
        # Volatility filter: ATR within reasonable bounds (0.3x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14_aligned[i] > 0.3 * atr_median[i]) and (atr_14_aligned[i] < 3.0 * atr_median[i])
        
        # Volume filter: current 6h volume > 1.5x median 1d volume
        vol_filter_6h = volume[i] > 1.5 * vol_median_1d_aligned[i]
        
        # Long: price breaks above 1d Donchian upper + volume + volatility + 12h uptrend
        if (close[i] > high_20_aligned[i] and 
            vol_filter_6h and 
            vol_filter and
            close[i] > ema_50_12h_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 1d Donchian lower + volume + volatility + 12h downtrend
        elif (close[i] < low_20_aligned[i] and 
              vol_filter_6h and 
              vol_filter and
              close[i] < ema_50_12h_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price re-enters the 1d Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_1dDonchian20_12hEMA50_Vol1.5x"
timeframe = "6h"
leverage = 1.0