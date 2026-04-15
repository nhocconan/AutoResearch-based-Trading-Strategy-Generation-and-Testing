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
    
    # Weekly trend: EMA(21) on weekly close
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, weekly, weekly_ema21)
    
    # Daily ATR(14) for volatility filter
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema21_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Only trade when weekly trend is defined and volatility sufficient
        if weekly_ema21_aligned[i] <= 0 or atr_14d_aligned[i] <= (0.003 * close[i]):
            signals[i] = 0.0
            continue
            
        # Long: Close above weekly EMA21 + volume spike
        if (close[i] > weekly_ema21_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Close below weekly EMA21 + volume spike
        elif (close[i] < weekly_ema21_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < weekly_ema21_aligned[i] and signals[i-1] > 0) or \
             (close[i] > weekly_ema21_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyTrend_Volume_Filter"
timeframe = "1d"
leverage = 1.0