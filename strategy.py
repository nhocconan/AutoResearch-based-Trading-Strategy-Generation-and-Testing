#!/usr/bin/env python3
# 6h_ElderRay_Alligator_TrendFilter
# Hypothesis: Elder Ray (Bull/Bear Power) combined with Williams Alligator acts as a robust trend filter.
# Bull Power > 0 and Bear Power < 0 with price above Alligator teeth indicates strong uptrend.
# Bear Power < 0 and Bull Power > 0 with price below Alligator teeth indicates strong downtrend.
# Uses 1-day EMA13 for Alligator (Jaw, Teeth, Lip) and 13-period EMA for Elder Ray power calculation.
# Designed for low trade frequency (~20-40/year) to minimize fee drift. Works in both bull and bear markets
# by requiring strong alignment of price, momentum, and trend structure before entering.

name = "6h_ElderRay_Alligator_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Daily EMA13 for Alligator (Jaw, Teeth, Lip) ---
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema8 = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema5 = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Alligator lines: Jaw (13), Teeth (8), Lip (5)
    jaw = ema13
    teeth = ema8
    lips = ema5
    
    # Align Alligator lines to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # --- Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13 ---
    bull_power = high - ema13  # High minus EMA13 (using daily EMA13)
    bear_power = low - ema13   # Low minus EMA13
    
    # Align Elder Ray powers to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # --- Trend strength filter: ADX-like using EMA crossover (Teeth > Jaw = uptrend) ---
    # We'll use teeth > jaw as bullish trend, teeth < jaw as bearish trend
    bullish_trend = teeth_aligned > jaw_aligned
    bearish_trend = teeth_aligned < jaw_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above Teeth, bullish trend
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > teeth_aligned[i] and 
                bullish_trend[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, price below Teeth, bearish trend
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] > 0 and 
                  close[i] < teeth_aligned[i] and 
                  bearish_trend[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: Bear Power >= 0 OR price crosses below Teeth
                if bear_power_aligned[i] >= 0 or close[i] < teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bull Power <= 0 OR price crosses above Teeth
                if bull_power_aligned[i] <= 0 or close[i] > teeth_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals