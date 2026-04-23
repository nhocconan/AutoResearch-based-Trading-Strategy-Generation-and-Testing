#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 1h timeframe for precise entry timing, combined with
4h EMA34 trend filter to avoid counter-trend trades. Volume spike confirms breakout momentum.
Designed for 1h timeframe with session filter (08-20 UTC) to reduce noise trades.
Target: 15-37 trades/year per symbol (60-150 total over 4 years).
Uses discrete position sizing (0.20) to minimize fee churn.
Works in both bull and bear markets by following the 4h trend direction.
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
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 1h Camarilla pivot levels (R1, S1)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla levels for 1h timeframe
    range_1h = high_1h - low_1h
    camarilla_r1 = close_1h + (range_1h * 1.1 / 12)
    camarilla_s1 = close_1h - (range_1h * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (previous 1h bar values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s1)
    
    # Calculate 4h EMA34 for primary trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 4h EMA34
        trend_up = close[i] > ema_34_4h_aligned[i]
        trend_down = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend on 4h AND volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_up and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND downtrend on 4h AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and trend_down and volume_spike[i]:
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

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0