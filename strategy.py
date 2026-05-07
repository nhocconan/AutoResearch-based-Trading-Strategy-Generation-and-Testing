#!/usr/bin/env python3
name = "6h_12h_1d_ElderRay_BullPower_BearPower_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Elder Ray components on daily: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) and 12h uptrend
            if bull_power_aligned[i] > 0 and ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling pressure) and 12h downtrend
            elif bear_power_aligned[i] < 0 and ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or trend turns down
            if bull_power_aligned[i] <= 0 or ema_34_12h_aligned[i] <= ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or trend turns up
            if bear_power_aligned[i] >= 0 or ema_34_12h_aligned[i] >= ema_34_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray with 12h trend filter
# - Elder Ray (Bull Power/Bear Power) measures daily buying/selling pressure relative to EMA(13)
# - Long when Bull Power > 0 (bulls in control) AND 12h uptrend
# - Short when Bear Power < 0 (bears in control) AND 12h downtrend
# - Works in both bull and bear markets via trend filter alignment
# - Elder Ray identifies institutional accumulation/distribution at daily level
# - 12h trend filter ensures we trade with higher timeframe momentum
# - Exit when power shifts or trend changes
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Novel on 6h: combines daily institutional pressure with 12h trend
# - Avoids whipsaws by requiring both power and trend alignment
# - Focus on BTC/ETH as primary targets (works across market regimes)