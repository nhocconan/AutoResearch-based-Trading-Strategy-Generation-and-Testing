#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: For 1h timeframe, use 4h trend filter (EMA50) to determine direction, 
then enter on 1h break of daily Camarilla R1/S1 with volume confirmation.
Use 1h only for entry timing to limit trades. Target 15-37 trades/year.
Works in bull/bear via 4h EMA50 trend filter.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R1, S1 levels
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align Camarilla levels to 1h (available after daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 1.5 * 24-period average (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_4h_aligned[i]
        r1_level = R1_aligned[i]
        s1_level = S1_aligned[i]
        vol_confirm_val = vol_confirm[i]
        
        if position == 0:
            # Long: break above R1 with volume and uptrend (4h EMA50)
            if close[i] > r1_level and vol_confirm_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: break below S1 with volume and downtrend (4h EMA50)
            elif close[i] < s1_level and vol_confirm_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: close below S1 or trend turns down
            if close[i] < s1_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above R1 or trend turns up
            if close[i] > r1_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0