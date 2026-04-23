#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Uses tighter Camarilla levels (R1/S1) for fewer, higher-quality breakouts combined with 4h EMA trend filter.
Volume confirmation avoids false breakouts. Designed for 1h timeframe to capture short-term moves
in both bull/bear markets via trend filter. Target: 15-37 trades/year per symbol (60-150 total over 4 years).
Uses discrete position sizing (0.20) to minimize fee churn.
Session filter (08-20 UTC) to reduce noise trades.
"""

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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot levels (R1, S1, R2, S2)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 1:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla levels based on previous 1h bar
    # R1 = close + (high - low) * 1.0/12
    # S1 = close - (high - low) * 1.0/12
    # R2 = close + (high - low) * 1.0/6
    # S2 = close - (high - low) * 1.0/6
    range_1h = high_1h - low_1h
    camarilla_r1 = close_1h + range_1h * (1.0/12)
    camarilla_s1 = close_1h - range_1h * (1.0/12)
    camarilla_r2 = close_1h + range_1h * (1.0/6)
    camarilla_s1 = close_1h - range_1h * (1.0/12)  # S1 repeated for clarity
    
    # Align Camarilla levels to 1h timeframe (previous bar values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)  # S2 same as S1 for now
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 4h EMA50 = uptrend, close < 4h EMA50 = downtrend
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend AND volume confirmation
            if close[i] > camarilla_r1_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.20
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0