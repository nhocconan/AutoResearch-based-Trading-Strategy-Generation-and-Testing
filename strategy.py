#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week trend filter.
# Long when: Bull Power > 0, Bear Power < 0, and 1-week EMA(50) rising.
# Short when: Bear Power < 0, Bull Power > 0, and 1-week EMA(50) falling.
# Exit when Elder Power signals weaken or reverse.
# Designed for ~15-30 trades/year per symbol. Works in bull/bear via trend filter.
name = "6h_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if EMA(50) weekly not ready
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: rising if current > previous
        if i == start_idx:
            weekly_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            weekly_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        
        bp = bull_power[i]
        bp_bear = bear_power[i]
        
        if position == 0:
            # Long: Bull Power positive, Bear Power negative, weekly up
            if bp > 0 and bp_bear < 0 and weekly_up:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, Bull Power positive, weekly down
            elif bp_bear < 0 and bp > 0 and weekly_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or Bear Power turns positive
            if bp <= 0 or bp_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive or Bull Power turns negative
            if bp_bear >= 0 or bp <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals