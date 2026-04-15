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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Donchian(20) for breakout structure
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 6h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(ema50_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14_aligned[i] > 0.5 * atr_median[i]) and (atr_14_aligned[i] < 3.0 * atr_median[i])
        
        # Long: price > daily EMA50 + Donchian breakout up + volume spike + volatility filter
        if (close[i] > ema50_aligned[i] and 
            close[i] > donch_high_20[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: price < daily EMA50 + Donchian breakout down + volume spike + volatility filter
        elif (close[i] < ema50_aligned[i] and 
              close[i] < donch_low_20[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high_20[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low_20[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_DailyEMA50_Donchian20_Vol1.5x_ATR14Filter"
timeframe = "6h"
leverage = 1.0