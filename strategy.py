#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Filtered_Close"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    ema50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily volume average for volume filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_weekly_aligned[i]
        vol_ok = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: close above weekly EMA50 with volume
            if close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: close below weekly EMA50 with volume
            elif close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly EMA50
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly EMA50
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals