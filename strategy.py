#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + 0.25 * (prev_high - prev_low)
    S1 = prev_close - 0.25 * (prev_high - prev_low)
    
    # Align to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 20-period EMA on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        ema_1w = ema_20_1w_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Close > R1 AND price > 1w EMA20 (uptrend) AND volume > 2.0x average
            if close[i] > r1 and close[i] > ema_1w and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S1 AND price < 1w EMA20 (downtrend) AND volume > 2.0x average
            elif close[i] < s1 and close[i] < ema_1w and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S1 OR trend reverses (price < 1w EMA20)
            if close[i] < s1 or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R1 OR trend reverses (price > 1w EMA20)
            if close[i] > r1 or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals