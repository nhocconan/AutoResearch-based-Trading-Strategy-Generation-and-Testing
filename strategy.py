#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
Camarilla pivot levels identify intraday support/resistance. Breakout at R1/S1 with 12h EMA trend
and volume confirmation avoids false signals. Designed for 4h timeframe to capture medium-term moves
in both bull/bear markets via trend filter. Target: 19-50 trades/year per symbol (75-200 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate 4h Camarilla pivot levels (R1, S1, R2, S2)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels based on previous 4h bar
    # R1 = close + (high - low) * 1.0/12
    # S1 = close - (high - low) * 1.0/12
    # R2 = close + (high - low) * 1.0/6
    # S2 = close - (high - low) * 1.0/6
    range_4h = high_4h - low_4h
    camarilla_r1 = close_4h + range_4h * (1.0/12)
    camarilla_s1 = close_4h - range_4h * (1.0/12)
    camarilla_r2 = close_4h + range_4h * (1.0/6)
    camarilla_s1 = close_4h - range_4h * (1.0/6)  # Note: S1 overwritten intentionally for tighter stop
    
    # Align Camarilla levels to 4h timeframe (previous bar values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)  # Use S1 for S2 alignment
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 12h EMA50 = uptrend, close < 12h EMA50 = downtrend
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend AND volume confirmation
            if close[i] > camarilla_r1_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S1 for longs, R1 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S1
                if close[i] < camarilla_s1_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R1
                if close[i] > camarilla_r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0