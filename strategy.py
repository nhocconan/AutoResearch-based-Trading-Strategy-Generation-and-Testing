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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA10 for trend
    ema10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Daily close for signal
    # Daily volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA10 to daily
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema10_1w_aligned[i]
        vol_ok = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: close above weekly EMA10 with volume
            if close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: close below weekly EMA10 with volume
            elif close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly EMA10
            if close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly EMA10
            if close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals