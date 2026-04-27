#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h only when 12h EMA50 confirms trend and volume > 1.5x 20-bar average. Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull (breakouts with trend) and bear (avoid false breakouts in ranging/weak trend) markets by requiring strong trend confirmation from HTF.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1 and S1 only for breakout)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    R1 = PP + range_1d * 1.0 / 4.0
    S1 = PP - range_1d * 1.0 / 4.0
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout entry with trend and volume confirmation
            long_breakout = close_val > R1_aligned[i]
            short_breakout = close_val < S1_aligned[i]
            trend_up = close_val > ema_50_12h_aligned[i]
            trend_down = close_val < ema_50_12h_aligned[i]
            
            if long_breakout and trend_up and volume_confirm[i]:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and volume_confirm[i]:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit on retracement to S1 or trend reversal
            if close_val < S1_aligned[i] or close_val < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on retracement to R1 or trend reversal
            if close_val > R1_aligned[i] or close_val > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0