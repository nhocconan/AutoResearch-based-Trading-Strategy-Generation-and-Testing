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
    
    # 1d EMA(20) for trend filter
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    ema_20d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20d_aligned = align_htf_to_ltf(prices, daily, ema_20d)
    
    # 1d ATR(14) for volatility filter
    high_d = daily['high'].values
    low_d = daily['low'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 2.0x median of last 30 bars (more selective)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median()
    vol_threshold = 2.0 * vol_median
    
    # ATR median for volatility regime filter (longer window for stability)
    atr_median = pd.Series(atr_14d_aligned).rolling(window=100, min_periods=100).median()
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20d_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(atr_median[i])):
            continue
        
        # Volatility filter: avoid extremes (0.7x to 2.5x of median ATR)
        vol_filter = (atr_14d_aligned[i] > 0.7 * atr_median[i]) and (atr_14d_aligned[i] < 2.5 * atr_median[i])
        
        # Long: Price above daily EMA20 + volume spike + volatility filter
        if (close[i] > ema_20d_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.30
        
        # Short: Price below daily EMA20 + volume spike + volatility filter
        elif (close[i] < ema_20d_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.30
        
        # Exit: price crosses back below/above daily EMA20
        elif (i > 0 and 
              ((signals[i-1] == 0.30 and close[i] < ema_20d_aligned[i]) or
               (signals[i-1] == -0.30 and close[i] > ema_20d_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyEMA20_Vol2.0x_ATR14dFilter_v3"
timeframe = "4h"
leverage = 1.0