# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_WilliamsAlligator_ElderRay_Trend
Hypothesis: Combine Williams Alligator (trend) and Elder Ray (bull/bear power) on daily timeframe with 1-week trend filter.
Go long when Alligator is bullish (jaws < teeth < lips) and Bull Power > 0 with weekly uptrend.
Go short when Alligator is bearish (jaws > teeth > lips) and Bear Power < 0 with weekly downtrend.
Exit when Alligator direction changes or Elder Ray power reverses.
Designed for low-frequency trend following (target 10-25 trades/year) to avoid fee drag and work in both bull and bear markets.
"""

name = "1d_WilliamsAlligator_ElderRay_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length <= 0:
        return source.copy()
    smma = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) == 0:
        return smma
    smma[0] = source[0]
    for i in range(1, len(source)):
        smma[i] = (smma[i-1] * (length-1) + source[i]) / length
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on daily: SMMA(13,8), SMMA(8,5), SMMA(5,3) - all shifted forward
    jaws_length = 13
    teeth_length = 8
    lips_length = 5
    
    # Calculate SMMA for median price
    median_price = (high + low) / 2
    smma_jaws = calculate_smma(median_price, jaws_length)
    smma_teeth = calculate_smma(median_price, teeth_length)
    smma_lips = calculate_smma(median_price, lips_length)
    
    # Shift forward as per Alligator definition
    jaws = np.roll(smma_jaws, -jaws_length//2)
    teeth = np.roll(smma_teeth, -teeth_length//2)
    lips = np.roll(smma_lips, -lips_length//2)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    weekly_uptrend = close_1w_aligned > ema34_1w_aligned
    weekly_downtrend = close_1w_aligned < ema34_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period: max of Alligator lengths and EMA13
    start_idx = max(jaws_length, teeth_length, lips_length, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        alligator_bullish = jaws[i] < teeth[i] and teeth[i] < lips[i]
        alligator_bearish = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        if position == 0:
            # Long: Alligator bullish + Bull Power positive + weekly uptrend
            if alligator_bullish and bull_power_positive and weekly_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Bear Power negative + weekly downtrend
            elif alligator_bearish and bear_power_negative and weekly_downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power becomes negative
            if not alligator_bullish or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power becomes positive
            if not alligator_bearish or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals