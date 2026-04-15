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
    
    # Weekly EMA(50) for trend filter
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    ema_50w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, weekly, ema_50w)
    
    # Weekly ATR(14) for volatility filter
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    tr1 = np.maximum(high_w[1:] - low_w[1:], np.abs(high_w[1:] - close_w[:-1]))
    tr2 = np.maximum(np.abs(low_w[1:] - close_w[:-1]), tr1)
    tr_w = np.concatenate([[np.nan], tr2])
    atr_14w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14w_aligned = align_htf_to_ltf(prices, weekly, atr_14w)
    
    # Daily volume median for volume filter
    daily = get_htf_data(prices, '1d')
    vol_d = daily['volume'].values
    vol_median_d = pd.Series(vol_d).rolling(window=20, min_periods=20).median()
    vol_median_d_aligned = align_htf_to_ltf(prices, daily, vol_median_d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50w_aligned[i]) or np.isnan(atr_14w_aligned[i]) or 
            np.isnan(vol_median_d_aligned[i])):
            continue
        
        # Volatility filter: avoid low volatility (ATR > 0.5x weekly ATR)
        vol_filter = atr_14w_aligned[i] > 0.5 * np.nanmedian(atr_14w_aligned[max(0, i-50):i+1])
        
        # Long: Price above weekly EMA50 + volume spike + volatility filter
        if (close[i] > ema_50w_aligned[i] and 
            volume[i] > 1.5 * vol_median_d_aligned[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price below weekly EMA50 + volume spike + volatility filter
        elif (close[i] < ema_50w_aligned[i] and 
              volume[i] > 1.5 * vol_median_d_aligned[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above weekly EMA50
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_50w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_50w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyEMA50_Vol1.5x_ATR14wFilter_v1"
timeframe = "1d"
leverage = 1.0