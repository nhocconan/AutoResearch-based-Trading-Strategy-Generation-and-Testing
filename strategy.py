#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_Volume_Session
Hypothesis: Price breaking above/below daily Camarilla R1/S1 on 1h timeframe with 4h EMA34 trend filter and volume spike (1.5x average) captures institutional breakouts. Designed for 1h to balance trade frequency and performance, using 4h for trend direction to avoid false signals. Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year.
"""
name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (daily levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    r1 = pivot + 1.1 * (prev_high - prev_low) / 12
    s1 = pivot - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume filter: current volume > 1.5 * 24-period average (balanced)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume spike + session
            if close[i] > r1_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume spike + session
            elif close[i] < s1_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (mean reversion)
            if position == 1:
                if close[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals