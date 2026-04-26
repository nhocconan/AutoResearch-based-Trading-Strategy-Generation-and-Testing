#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_v2
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts from prior 1h bar with 4h EMA50 trend filter. Target 15-35 trades/year by requiring confluence of HTF trend alignment and price structure breakout. Uses discrete position sizing (0.20) and time-based exits to control trade frequency and fee drag. Designed to work in both bull and bear markets via trend filter that adapts to market direction.
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
    
    # Get 4h data for HTF trend (EMA50) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1h data for Camarilla levels - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1h bar (HLC of prior 1h)
    cam_high = pd.Series(df_1h['high'].values).shift(1).values
    cam_low = pd.Series(df_1h['low'].values).shift(1).values
    cam_close = pd.Series(df_1h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    R1_aligned = align_htf_to_ltf(prices, df_1h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0
    
    # Warmup: max of EMA(50) 4h, Camarilla (need 2 bars for shift)
    start_idx = max(50, 2) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        if position == 0:
            # Long: break above R1 with uptrend
            long_signal = (close_val > r1_val) and uptrend
            
            # Short: break below S1 with downtrend
            short_signal = (close_val < s1_val) and downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                bars_in_trade = 0
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                bars_in_trade = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            bars_in_trade += 1
            # Time-based exit: exit after 24 hours (24 bars on 1h)
            if bars_in_trade >= 24:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            bars_in_trade += 1
            # Time-based exit: exit after 24 hours (24 bars on 1h)
            if bars_in_trade >= 24:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_v2"
timeframe = "1h"
leverage = 1.0