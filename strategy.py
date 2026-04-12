#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_elder_ray_regime_v1
# Uses Elder Ray (Bull/Bear Power) from daily timeframe to measure bull/bear strength.
# Combines with 12h trend (EMA21 > EMA50) and volume confirmation for entries.
# Long when Bull Power > 0, Bear Power < 0, and 12h EMA21 > EMA50 with volume > 1.5x 20-bar avg.
# Short when Bull Power < 0, Bear Power > 0, and 12h EMA21 < EMA50 with volume confirmation.
# Exits when Elder Ray signals weaken or reverse.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via trend alignment and in ranging markets via mean reversion to zero.

name = "6h_1d_1w_elder_ray_regime_v1"
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
    
    # Get daily data for Elder Ray calculation (requires EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high - ema13_1d  # Bear Power uses low, Bull Power uses high
    bear_power_1d = low - ema13_1d
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 12h data for trend filter (EMA21 > EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema21_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power > 0, Bear Power < 0, and 12h EMA21 > EMA50
        long_condition = (bull_power_aligned[i] > 0 and 
                         bear_power_aligned[i] < 0 and
                         ema21_12h_aligned[i] > ema50_12h_aligned[i])
        
        # Short conditions: Bull Power < 0, Bear Power > 0, and 12h EMA21 < EMA50
        short_condition = (bull_power_aligned[i] < 0 and 
                          bear_power_aligned[i] > 0 and
                          ema21_12h_aligned[i] < ema50_12h_aligned[i])
        
        # Entry signals
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: Elder Ray signals weaken or reverse
        elif position == 1 and (bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power_aligned[i] >= 0 or bear_power_aligned[i] <= 0):
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