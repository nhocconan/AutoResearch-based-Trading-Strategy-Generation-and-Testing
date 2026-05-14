#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_elder_ray_v1
# Uses daily Elder Ray (bull/bear power) to identify trend strength and reversal points.
# Long when daily bull power > 0 and price > EMA13 on 6h (bullish momentum).
# Short when daily bear power < 0 and price < EMA13 on 6h (bearish momentum).
# Uses volume confirmation (volume > 1.5x 20-period average) to filter false signals.
# Designed for low trade frequency (target: 15-40 trades/year) to minimize fee drag.
# Works in bull markets (buying strength) and bear markets (selling weakness).

name = "6h_1d_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA13 for daily close (used in Elder Ray)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 6h timeframe (daily values update after daily bar closes)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate EMA13 on 6h for entry timing
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema13_6h[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: daily bull power positive AND price above 6h EMA13
        if bull_power_aligned[i] > 0 and close[i] > ema13_6h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: daily bear power negative AND price below 6h EMA13
        elif bear_power_aligned[i] < 0 and close[i] < ema13_6h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite Elder Ray signal
        elif bull_power_aligned[i] <= 0 and position == 1:
            position = 0
            signals[i] = 0.0
        elif bear_power_aligned[i] >= 0 and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals