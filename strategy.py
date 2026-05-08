#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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
    
    # Get 4h data for Camarilla and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (previous day)
    high_prev = df_4h['high'].shift(1).values
    low_prev = df_4h['low'].shift(1).values
    close_prev = df_4h['close'].shift(1).values
    
    # True range for Camarilla
    tr = np.maximum(high_prev - low_prev, 
                    np.maximum(np.abs(high_prev - close_prev), 
                               np.abs(low_prev - close_prev)))
    
    # Camarilla levels
    R1 = close_prev + (tr * 1.1 / 12)
    S1 = close_prev - (tr * 1.1 / 12)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume ratio (current vs 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_20 > 0, vol_ma_20, 1)
    
    # Align all indicators to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
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
            # Long: price breaks above R1 + above EMA50 + volume confirmation
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_50_aligned[i] and
                vol_ratio_1d_aligned[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + below EMA50 + volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and
                  vol_ratio_1d_aligned[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR below EMA50
            if (close[i] < S1_aligned[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 OR above EMA50
            if (close[i] > R1_aligned[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals