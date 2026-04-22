#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R1, EMA50 trend up, and volume > 1.5x average.
Short when price breaks below S1, EMA50 trend down, and volume > 1.5x average.
Exit when price crosses back below R1 (long) or above S1 (short).
Camarilla levels provide high-probability reversal points; EMA50 filters trend direction;
volume spike confirms institutional participation. Works in bull/bear via trend filter.
"""

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
    
    # Load 1-day data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    pivot = (ph + pl + pc) / 3.0
    r1 = pc + (ph - pl) * 1.1 / 12.0
    s1 = pc - (ph - pl) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate average volume for volume spike filter
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, EMA50 trending up, volume spike
            if close[i] > r1_aligned[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, EMA50 trending down, volume spike
            elif close[i] < s1_aligned[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses back below R1
                if close[i] < r1_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses back above S1
                if close[i] > s1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0