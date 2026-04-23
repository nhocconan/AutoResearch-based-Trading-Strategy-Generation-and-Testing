#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Uses tighter Camarilla levels (R3/S3) for fewer, higher-quality breakouts combined with 1d EMA trend filter.
Volume confirmation avoids false breakouts. Designed for 6h timeframe to capture medium-term moves
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Camarilla pivot levels (R3, S3, R4, S4)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 1:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla levels based on previous 6h bar
    # R3 = close + (high - low) * 1.0/4
    # S3 = close - (high - low) * 1.0/4
    # R4 = close + (high - low) * 1.0/2
    # S4 = close - (high - low) * 1.0/2
    range_6h = high_6h - low_6h
    camarilla_r3 = close_6h + range_6h * (1.0/4)
    camarilla_s3 = close_6h - range_6h * (1.0/4)
    camarilla_r4 = close_6h + range_6h * (1.0/2)
    camarilla_s4 = close_6h - range_6h * (1.0/2)
    
    # Align Camarilla levels to 6h timeframe (previous bar values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s4)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S3 for longs, R3 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S3
                if close[i] < camarilla_s3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0