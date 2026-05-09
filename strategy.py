#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_ZeroCross_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX, trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of log returns)
    close_1d = df_1d['close'].values
    log_returns = np.log(close_1d[1:] / close_1d[:-1])
    log_returns = np.concatenate([[np.nan], log_returns])  # align with original index
    
    ema1 = pd.Series(log_returns).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Align TRIX to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need 30 for TRIX, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND price > 1d EMA50 (uptrend) AND volume > 2.0x average
            if trix_val > 0 and trix_aligned[i-1] <= 0 and close[i] > ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND price < 1d EMA50 (downtrend) AND volume > 2.0x average
            elif trix_val < 0 and trix_aligned[i-1] >= 0 and close[i] < ema_1d and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reverses (price < 1d EMA50)
            if trix_val < 0 and trix_aligned[i-1] >= 0 or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reverses (price > 1d EMA50)
            if trix_val > 0 and trix_aligned[i-1] <= 0 or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals