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
    
    # Load daily high/low for pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily pivot point (classic: (H+L+C)/3)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    # Calculate support/resistance levels
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Align pivot levels to 4h timeframe (properly delayed)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price above S1 and EMA50, with volume confirmation
        if (close[i] > s1_aligned[i] and close[i] > ema_50[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price below R1 and below EMA50, with volume confirmation
        elif (close[i] < r1_aligned[i] and close[i] < ema_50[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back through pivot or EMA50
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < pivot_aligned[i] or close[i] < ema_50[i])) or
               (signals[i-1] == -0.25 and (close[i] > pivot_aligned[i] or close[i] > ema_50[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyPivot_EMA_Volume_Filter"
timeframe = "4h"
leverage = 1.0