#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray power with 1w trend filter
# Uses Jaw/Teeth/Lips for trend direction, Bull/Bear power for entry timing.
# 1w EMA50 filter ensures we only trade with the weekly trend to avoid counter-trend whipsaws.
# Discrete sizing 0.25 to limit fee drag. Target: 60-120 total trades over 4 years (15-30/year).
# Alligator identifies trendless markets (all lines intertwined) - we avoid those.
# Elder Ray power confirms bull/bear energy behind moves.

name = "6h_WilliamsAlligator_ElderRay_1wEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with offsets)
    # Jaw: 13-period SMA, offset 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, offset 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, offset 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull power: high - EMA13
    bear_power = low - ema13   # Bear power: low - EMA13
    
    # Calculate 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 8, 5)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Alligator alignment: check if lines are separated (trending market)
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Jaw > Teeth > Lips
        bullish_aligned = curr_lips > curr_teeth and curr_teeth > curr_jaw
        bearish_aligned = curr_jaw > curr_teeth and curr_teeth > curr_lips
        
        if position == 0:  # Flat - look for new entries
            # Only trade with weekly trend filter and Alligator alignment
            if bullish_aligned and curr_close > curr_ema_50_1w:
                # Long: Bull power positive and increasing
                if curr_bull_power > 0 and curr_bull_power > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            elif bearish_aligned and curr_close < curr_ema_50_1w:
                # Short: Bear power negative and decreasing
                if curr_bear_power < 0 and curr_bear_power < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses OR bear power becomes strong
            if not bullish_aligned or curr_bear_power < -0.5 * np.std(bear_power[max(0,i-50):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses OR bull power becomes strong
            if not bearish_aligned or curr_bull_power > 0.5 * np.std(bull_power[max(0,i-50):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals