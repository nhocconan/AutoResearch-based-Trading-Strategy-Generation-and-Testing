#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation.
Target: 20-50 trades/year per symbol. Uses discrete position sizing (0.30) to minimize fee churn.
Works in both bull/bear via trend filter and volume confirmation to avoid false breakouts.
Camarilla R4/S4 levels provide stronger breakout signals than R3/S3, reducing whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Camarilla levels (using previous 4h bar's range)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R4, S4 levels: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    range_4h = high_4h - low_4h
    camarilla_r4_4h = close_4h + range_4h * 1.1 / 2
    camarilla_s4_4h = close_4h - range_4h * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 12h EMA50 = uptrend, close < 12h EMA50 = downtrend
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (stricter to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R4 AND uptrend AND volume confirmation
            if close[i] > camarilla_r4_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.30
                position = 1
            # Short: Break below Camarilla S4 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s4_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: reverse signal or break of opposite Camarilla level
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S4
                if close[i] < camarilla_s4_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R4
                if close[i] > camarilla_r4_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0