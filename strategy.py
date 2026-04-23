#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 1d Camarilla R1 AND 1w EMA50 uptrend AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S1 AND 1w EMA50 downtrend AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for longs, R1 for shorts).
Uses 1w HTF for EMA50 trend strength (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
Camarilla levels provide precise intraday support/resistance; EMA50 filter ensures we only trade with the weekly trend.
Works in both bull and bear markets by following the 1w trend direction.
"""

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
    
    # Calculate 1d Camarilla levels (based on previous day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    rang = high_1d - low_1d
    camarilla_r1 = close_1d + rang * 1.1 / 12
    camarilla_s1 = close_1d - rang * 1.1 / 12
    
    # Align 1d Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # volume MA (20), Camarilla needs at least 2 days
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND 1w EMA50 uptrend AND volume spike
            if price > r1 and close[i] > ema_50 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND 1w EMA50 downtrend AND volume spike
            elif price < s1 and close[i] < ema_50 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s1:  # Long exit at Camarilla S1
                exit_signal = True
            elif position == -1 and price > r1:  # Short exit at Camarilla R1
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeConfirmation_LevelExit"
timeframe = "12h"
leverage = 1.0