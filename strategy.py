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
    
    # 1d daily data
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    high_d = daily['high'].values
    low_d = daily['low'].values
    volume_d = daily['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, daily, ema_50d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 1d volume SMA(20)
    vol_sma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    vol_sma_d_aligned = align_htf_to_ltf(prices, daily, vol_sma_d)
    
    # 6h ATR(14) for position sizing and volatility filter
    tr1_6h = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2_6h = np.maximum(np.abs(low[1:] - close[:-1]), tr1_6h)
    tr_6h = np.concatenate([[np.nan], tr2_6h])
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR median for volatility regime filter
    atr_median = pd.Series(atr_14_6h).rolling(window=50, min_periods=50).median()
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50d_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(vol_sma_d_aligned[i]) or np.isnan(atr_median[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        vol_filter = (atr_14_6h[i] > 0.5 * atr_median[i]) and (atr_14_6h[i] < 3.0 * atr_median[i])
        
        # Volume filter: 1.5x daily average volume
        vol_filter_6h = volume[i] > 1.5 * vol_sma_d_aligned[i]
        
        # Long: Price above daily EMA50 + volume spike + volatility filter
        if (close[i] > ema_50d_aligned[i] and 
            vol_filter_6h and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price below daily EMA50 + volume spike + volatility filter
        elif (close[i] < ema_50d_aligned[i] and 
              vol_filter_6h and 
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

name = "6h_DailyEMA50_Vol1.5x_ATR14dFilter"
timeframe = "6h"
leverage = 1.0