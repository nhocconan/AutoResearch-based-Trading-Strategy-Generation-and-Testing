#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA200 trend filter and volume confirmation.
Uses primary Camarilla levels (R1/S1) for fewer, higher-quality breakouts combined with 1d EMA200 trend filter.
Volume confirmation avoids false breakouts. Designed for 12h timeframe to capture medium-term moves
in both bull/bear markets via trend filter. Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 12h Camarilla pivot levels (R1, S1, R2, S2)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels based on previous 12h bar
    # R1 = close + (high - low) * 1.0/12
    # S1 = close - (high - low) * 1.0/12
    # R2 = close + (high - low) * 1.0/6
    # S2 = close - (high - low) * 1.0/6
    range_12h = high_12h - low_12h
    camarilla_r1 = close_12h + range_12h * (1.0/12)
    camarilla_s1 = close_12h - range_12h * (1.0/12)
    camarilla_r2 = close_12h + range_12h * (1.0/6)
    camarilla_s1 = close_12h - range_12h * (1.0/12)  # Recalculate S1 to avoid overwrite
    camarilla_s2 = close_12h - range_12h * (1.0/6)
    
    # Align Camarilla levels to 12h timeframe (previous bar values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s2)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # need EMA200 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA200 = uptrend, close < 1d EMA200 = downtrend
        trend_up = close[i] > ema_200_1d_aligned[i]
        trend_down = close[i] < ema_200_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
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

name = "12H_Camarilla_R1S1_Breakout_1dEMA200_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0