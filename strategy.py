#!/usr/bin/env python3
"""
12h_Williams_Alligator_ElderRay
Hypothesis: Combines Williams Alligator (trend detection) and Elder Ray (bull/bear power) on 12h timeframe with weekly trend filter.
Williams Alligator identifies trend phases using smoothed medians, Elder Ray measures bull/bear power relative to EMA13.
Weekly trend filter ensures alignment with higher timeframe momentum. Designed for low trade frequency with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMMA(median price, 13, 8, 5)
    median_price = (high + low) / 2
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = np.full_like(close, np.nan)
    if len(close) >= 13:
        ema13[12] = np.mean(close[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, len(close)):
            ema13[i] = close[i] * alpha + ema13[i-1] * (1 - alpha)
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 34)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_up = lips[i] > teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Alligator uptrend + Bull Power > 0 + weekly uptrend
            if alligator_up and bull_power[i] > 0 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Bear Power < 0 + weekly downtrend
            elif alligator_down and bear_power[i] < 0 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns down OR Bull Power <= 0
            if not alligator_up or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns up OR Bear Power >= 0
            if not alligator_down or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_ElderRay"
timeframe = "12h"
leverage = 1.0