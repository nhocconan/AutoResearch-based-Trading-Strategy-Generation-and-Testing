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
    
    # 1d EMA(50) for trend filter
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    ema_50d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, daily, ema_50d)
    
    # 1d ATR(14) for volatility regime filter
    high_d = daily['high'].values
    low_d = daily['low'].values
    tr = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 1d volatility median for regime filter
    atr_median = pd.Series(atr_14d_aligned).rolling(window=100, min_periods=100).median()
    
    # 6h Donchian(20) breakout levels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume threshold: 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50d_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(atr_median[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        vol_filter = (atr_14d_aligned[i] > 0.5 * atr_median[i]) and (atr_14d_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Price breaks above Donchian high + above daily EMA50 + volume spike + volatility filter
        if (close[i] > donch_high[i] and 
            close[i] > ema_50d_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian low + below daily EMA50 + volume spike + volatility filter
        elif (close[i] < donch_low[i] and 
              close[i] < ema_50d_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above daily EMA50
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_50d_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_50d_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Donchian20_EMA50_Vol1.5x_ATR14dFilter_v1"
timeframe = "6h"
leverage = 1.0