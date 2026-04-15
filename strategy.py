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
    
    # 12h EMA(20) for trend filter (HTF)
    daily_12h = get_htf_data(prices, '12h')
    close_12h = daily_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, daily_12h, ema_20_12h)
    
    # 12h ATR(14) for volatility filter (HTF)
    high_12h = daily_12h['high'].values
    low_12h = daily_12h['low'].values
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.maximum(np.abs(low_12h[1:] - close_12h[:-1]), tr1)
    tr_12h = np.concatenate([[np.nan], tr2])
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, daily_12h, atr_14_12h)
    
    # Volume threshold: 2.0x median of last 20 bars (more selective)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    # Volatility regime filter: avoid extremes (0.3x to 2.5x of median ATR)
    atr_median = pd.Series(atr_14_12h_aligned).rolling(window=30, min_periods=30).median()
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(atr_median[i])):
            continue
        
        # Volatility filter: avoid extremes (0.3x to 2.5x of median ATR)
        vol_filter = (atr_14_12h_aligned[i] > 0.3 * atr_median[i]) and (atr_14_12h_aligned[i] < 2.5 * atr_median[i])
        
        # Long: Price above 12h EMA20 + volume spike + volatility filter
        if (close[i] > ema_20_12h_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price below 12h EMA20 + volume spike + volatility filter
        elif (close[i] < ema_20_12h_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above 12h EMA20
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_20_12h_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_20_12h_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_12hEMA20_Vol2.0x_ATR14dFilter_v1"
timeframe = "4h"
leverage = 1.0