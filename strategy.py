#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_TRIX_Zero_Cross_4hTrend_Volume_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and TRIX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close (triple EMA)
    close_4h = df_4h['close'].values
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (ema3 - prev_ema3) / prev_ema3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / np.where(np.abs(ema3[:-1]) > 1e-10, np.abs(ema3[:-1]), 1e-10) * 100
    trix_raw[0] = 0.0
    
    # Align TRIX to 1h
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix_raw)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Daily volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume ratio: current 1h volume vs 20-day average volume (scaled)
    vol_ratio = volume / np.where(vol_avg_1d_aligned > 0, vol_avg_1d_aligned / 24.0, 1.0)  # 24 hours in day
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + above 4h EMA34 + volume confirmation
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and  # zero cross up
                close[i] > ema_34_4h_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.20
                position = 1
            # Short: TRIX crosses below zero + below 4h EMA34 + volume confirmation
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and  # zero cross down
                  close[i] < ema_34_4h_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR price below EMA34
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: TRIX crosses above zero OR price above EMA34
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals