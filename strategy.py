#!/usr/bin/env python3
"""
1d_Aroon_Trend_Strength_v1
Hypothesis: Aroon oscillator from 1d measures trend strength and direction.
When Aroon Up > 70 (strong uptrend) go long, Aroon Down > 70 (strong downtrend) go short.
Weak trend (both < 30) triggers exit. Combined with 1w trend filter to avoid counter-trend trades.
Target: 20-60 total trades over 4 years on 1d timeframe.
"""

name = "1d_Aroon_Trend_Strength_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Aroon ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need at least 25 days for Aroon(25)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Aroon Up: ((25 - periods since 25-period high) / 25) * 100
    # Aroon Down: ((25 - periods since 25-period low) / 25) * 100
    period = 25
    high_idx = pd.Series(high_1d).rolling(window=period, min_periods=1).apply(lambda x: np.argmax(x), raw=True)
    low_idx = pd.Series(low_1d).rolling(window=period, min_periods=1).apply(lambda x: np.argmin(x), raw=True)
    
    # Convert index to periods ago
    periods_since_high = (period - 1) - high_idx
    periods_since_low = (period - 1) - low_idx
    
    aroon_up = ((period - periods_since_high) / period) * 100
    aroon_down = ((period - periods_since_low) / period) * 100
    
    # === 1W Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple trend: price above/below 20-period EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align Aroon to 1d timeframe
    aroon_up_aligned = align_htf_to_ltf(prices, df_1d, aroon_up)
    aroon_down_aligned = align_htf_to_ltf(prices, df_1d, aroon_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(aroon_up_aligned[i]) or 
            np.isnan(aroon_down_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Aroon signals
        strong_uptrend = aroon_up_aligned[i] > 70
        strong_downtrend = aroon_down_aligned[i] > 70
        weak_trend = (aroon_up_aligned[i] < 30) and (aroon_down_aligned[i] < 30)
        
        # 1W trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: strong uptrend + weekly uptrend
            if strong_uptrend and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend + weekly downtrend
            elif strong_downtrend and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weak trend OR weekly trend turns down
            if weak_trend or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: weak trend OR weekly trend turns up
            if weak_trend or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals