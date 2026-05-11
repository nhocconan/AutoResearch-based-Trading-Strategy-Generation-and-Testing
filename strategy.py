#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_Trend_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla levels (primary direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for previous 4h bar
    # H, L, C from previous completed 4h bar
    H = high_4h[:-1]  # shift by 1 to get previous bar
    L = low_4h[:-1]
    C = close_4h[:-1]
    
    # Camarilla levels
    R4 = C + ((H - L) * 1.1 / 2)
    R3 = C + ((H - L) * 1.1 / 4)
    R2 = C + ((H - L) * 1.1 / 6)
    R1 = C + ((H - L) * 1.1 / 12)
    S1 = C - ((H - L) * 1.1 / 12)
    S2 = C - ((H - L) * 1.1 / 6)
    S3 = C - ((H - L) * 1.1 / 4)
    S4 = C - ((H - L) * 1.1 / 2)
    
    # Align Camarilla levels to 1h
    R4_1h = align_htf_to_ltf(prices, df_4h, R4)
    R3_1h = align_htf_to_ltf(prices, df_4h, R3)
    R2_1h = align_htf_to_ltf(prices, df_4h, R2)
    R1_1h = align_htf_to_ltf(prices, df_4h, R1)
    S1_1h = align_htf_to_ltf(prices, df_4h, S1)
    S2_1h = align_htf_to_ltf(prices, df_4h, S2)
    S3_1h = align_htf_to_ltf(prices, df_4h, S3)
    S4_1h = align_htf_to_ltf(prices, df_4h, S4)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(R1_1h[i]) or np.isnan(S1_1h[i]) or 
            np.isnan(ema50_1h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check filters
        if not (session_filter[i] and volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > R1 and above EMA50 (bullish trend)
            if close[i] > R1_1h[i] and close[i] > ema50_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < S1 and below EMA50 (bearish trend)
            elif close[i] < S1_1h[i] and close[i] < ema50_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < S1 (strong bearish reversal) or below EMA50
            if close[i] < S1_1h[i] or close[i] < ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > R1 (strong bullish reversal) or above EMA50
            if close[i] > R1_1h[i] or close[i] > ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals