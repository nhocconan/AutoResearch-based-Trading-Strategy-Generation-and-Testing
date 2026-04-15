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
    open_time = prices['open_time']
    
    # Precompute hour filter (08-20 UTC) to reduce noise
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for trend filter (once before loop)
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    high_d = daily['high'].values
    low_d = daily['low'].values
    
    # EMA200 on daily for long-term trend
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_d_aligned = align_htf_to_ltf(prices, daily, ema200_d)
    
    # 4h data for entry timing
    h4 = get_htf_data(prices, '4h')
    close_4h = h4['close'].values
    high_4h = h4['high'].values
    low_4h = h4['low'].values
    volume_4h = h4['volume'].values
    
    # 4h EMA20 for short-term trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, h4, ema20_4h)
    
    # 4h ATR for volatility filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, h4, atr_14_4h)
    
    # Volume filter: 1.5x median of last 20 periods on 4h
    vol_median_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).median()
    vol_threshold_4h = 1.5 * vol_median_4h
    vol_4h_aligned = align_htf_to_ltf(prices, h4, vol_threshold_4h.values)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_d_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(vol_4h_aligned[i])):
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        if atr_14_4h_aligned[i] < (0.01 * close[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: require above-average volume
        if volume[i] < vol_4h_aligned[i]:
            signals[i] = 0.0
            continue
        
        # Long condition: price above daily EMA200 AND above 4h EMA20
        if (close[i] > ema200_d_aligned[i] and 
            close[i] > ema20_4h_aligned[i]):
            signals[i] = 0.20
        
        # Short condition: price below daily EMA200 AND below 4h EMA20
        elif (close[i] < ema200_d_aligned[i] and 
              close[i] < ema20_4h_aligned[i]):
            signals[i] = -0.20
        
        # Exit: reverse signal when trend condition fails
        elif (signals[i-1] > 0 and close[i] < ema20_4h_aligned[i]) or \
             (signals[i-1] < 0 and close[i] > ema20_4h_aligned[i]):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_EMA200_4hEMA20_Volume_Filter"
timeframe = "1h"
leverage = 1.0