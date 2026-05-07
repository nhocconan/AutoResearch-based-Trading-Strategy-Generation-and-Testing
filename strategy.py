#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    open_ = prices['open'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for volume average (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Camarilla levels for previous 1d (high, low, close)
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    # Camarilla R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h volume average (20-period) for RVOL
    vol_ma20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate RVOL (current 4h volume / average 4h volume over last 20)
        if vol_ma20_4h[i] > 0:
            rvol = volume[i] / vol_ma20_4h[i]
        else:
            rvol = 0
        
        if position == 0:
            # Long: price above 1d EMA50 (uptrend), price breaks above R1, volume spike (RVOL > 1.5)
            if (close[i] > ema_50_1d_aligned[i] and 
                close[i] > r1_aligned[i] and 
                rvol > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA50 (downtrend), price breaks below S1, volume spike (RVOL > 1.5)
            elif (close[i] < ema_50_1d_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  rvol > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals