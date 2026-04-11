#!/usr/bin/env python3
# 6h_1w_elder_ray_regime_v1
# Strategy: 6h Elder Ray (Bull/Bear Power) with weekly trend regime filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Elder Ray measures bull/bear power via EMA(13). Combined with weekly trend (EMA40),
# it avoids counter-trend trades. Long when Bull Power > 0 and weekly trend up; short when Bear Power < 0 and weekly trend down.
# Works in bull via trend-following longs, in bear via trend-following shorts. Low turnover expected.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6-day EMA13 for Elder Ray (using close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Weekly EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(ema40_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend: up if close > EMA40, down if close < EMA40
        weekly_up = close[i] > ema40_1w_aligned[i]
        weekly_down = close[i] < ema40_1w_aligned[i]
        
        # Entry conditions
        # Long: Bull Power > 0 AND weekly trend up
        if bull_power[i] > 0 and weekly_up and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bear Power < 0 AND weekly trend down
        elif bear_power[i] < 0 and weekly_down and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Elder Ray signal OR weekly trend reversal
        elif position == 1 and (bull_power[i] <= 0 or not weekly_up):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] >= 0 or not weekly_down):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals