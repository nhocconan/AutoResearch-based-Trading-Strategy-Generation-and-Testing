#!/usr/bin/env python3
name = "12h_Camarilla_Pivot_Volume_Trend_Filter"
timeframe = "12h"
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
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    
    # Daily volume filter: volume > 1.8x 20-day average (stricter to reduce trades)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.8 * vol_ma20_1d_aligned
    
    # Camarilla levels from previous 1d (H, L, C)
    # Calculate only once per day using previous day's OHLC
    # We'll compute H, L, C for each 1d bar, then align to 12h
    # For each 12h bar, we use the previous completed 1d bar's H, L, C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # We focus on R3, S3 for entries, R4, S4 for stops (but we'll use close-based exit)
    diff = high_1d - low_1d
    R3 = close_1d + (diff * 1.1 / 4)
    S3 = close_1d - (diff * 1.1 / 4)
    R4 = close_1d + (diff * 1.1 / 2)
    S4 = close_1d - (diff * 1.1 / 2)
    
    # Align to 12h - these levels are valid after the 1d bar closes
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above S3 (support hold) with uptrend and volume
            # Actually, we want to buy on strength: close above R3 with uptrend
            if close[i] > R3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below R3 (resistance hold) with downtrend and volume
            elif close[i] < S3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below S3 (support broken) or trend down
            if close[i] < S3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above R3 (resistance broken) or trend up
            if close[i] > R3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals