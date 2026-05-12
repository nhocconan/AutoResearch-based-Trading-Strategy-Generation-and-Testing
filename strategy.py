#!/usr/bin/env python3
# 6H_ELDER_RAY_REGIME
# Hypothesis: Elder Ray Index (Bull/Bear Power) combined with 1d trend regime filter to capture institutional sentiment shifts.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Go long when Bull Power turns positive with EMA13 rising and price above 1d EMA50.
# Go short when Bear Power turns positive with EMA13 falling and price below 1d EMA50.
# Uses 1d EMA50 as regime filter to avoid counter-trend trades. Designed for 6H timeframe to reduce noise and capture sustained moves.
# Targets 15-30 trades/year (~60-120 over 4 years) to minimize fee drag while capturing high-probability institutional moves.

name = "6H_ELDER_RAY_REGIME"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray calculation
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # EMA13 slope for trend confirmation
    ema13_prev = np.roll(ema13, 1)
    ema13_prev[0] = ema13[0]
    ema13_rising = ema13 > ema13_prev
    ema13_falling = ema13 < ema13_prev
    
    # 1d EMA50 for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema50_1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_rising[i]) or np.isnan(ema13_falling[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power turns positive AND EMA13 rising AND price above 1d EMA50 (bullish regime)
            if bull_power[i] > 0 and bull_power[i-1] <= 0 and ema13_rising[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power turns positive AND EMA13 falling AND price below 1d EMA50 (bearish regime)
            elif bear_power[i] > 0 and bear_power[i-1] <= 0 and ema13_falling[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR EMA13 turns flat/failing
            if bull_power[i] <= 0 or not ema13_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative OR EMA13 turns flat/rising
            if bear_power[i] <= 0 or not ema13_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals