#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly ATR (14-period) for volatility filter
    weekly = get_htf_data(prices, '1w')
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    close_w = weekly['close'].values
    tr1 = np.maximum(high_w[1:] - low_w[1:], np.abs(high_w[1:] - close_w[:-1]))
    tr2 = np.maximum(np.abs(low_w[1:] - close_w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14w_aligned = align_htf_to_ltf(prices, weekly, atr_14w)
    
    # Monthly Donchian channel (20-period)
    monthly = get_htf_data(prices, '1M')
    high_m = monthly['high'].values
    low_m = monthly['low'].values
    high_20m = pd.Series(high_m).rolling(window=20, min_periods=20).max().values
    low_20m = pd.Series(low_m).rolling(window=20, min_periods=20).min().values
    high_20m_aligned = align_htf_to_ltf(prices, monthly, high_20m)
    low_20m_aligned = align_htf_to_ltf(prices, monthly, low_20m)
    
    # Volume confirmation: current > 1.5x median of last 30 bars
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20m_aligned[i]) or np.isnan(low_20m_aligned[i]) or
            np.isnan(atr_14w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        atr_median = pd.Series(atr_14w_aligned).rolling(window=50, min_periods=50).median()
        vol_filter = (atr_14w_aligned[i] > 0.5 * atr_median[i]) and (atr_14w_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Monthly Donchian breakout up + volume spike + volatility filter
        if (close[i] > high_20m_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Monthly Donchian breakout down + volume spike + volatility filter
        elif (close[i] < low_20m_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters monthly Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20m_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20m_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_MonthlyDonchian20_Vol1.5x_ATR14wFilter"
timeframe = "1d"
leverage = 1.0