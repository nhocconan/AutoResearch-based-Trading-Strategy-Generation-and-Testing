#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wTrendFilter_v1
Hypothesis: Trade Ichimoku Tenkan-Kijun cross on 6h with 1w EMA50 trend filter. TK cross provides timely momentum signals while weekly EMA50 ensures alignment with major trend, reducing whipsaws in ranging/bear markets. Discrete sizing (0.25) limits fee drag. Target: 12-30 trades/year per symbol to survive 2021-2026 regimes.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (26) and 1w EMA50 (50)
    start_idx = max(26, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: Tenkan crosses above Kijun + 1w uptrend
            tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            
            # Short setup: Tenkan crosses below Kijun + 1w downtrend
            tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            
            if tk_cross_up and htf_1w_bullish:
                signals[i] = 0.25
                position = 1
            elif tk_cross_down and htf_1w_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun (trend weakening) OR 1w trend turns bearish
            tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
            if tk_cross_down or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun (trend weakening) OR 1w trend turns bullish
            tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            if tk_cross_up or htf_1w_bullish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wTrendFilter_v1"
timeframe = "6h"
leverage = 1.0