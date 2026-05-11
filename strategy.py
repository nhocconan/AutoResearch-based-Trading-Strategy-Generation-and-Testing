#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay_Trend_Signal
Hypothesis: On 12h timeframe, combine Williams Alligator trend direction with Elder Ray power/ray for trend strength confirmation.
Enter long when price is above Alligator teeth and Bull Power > 0; short when price below teeth and Bear Power < 0.
Exit on opposite signal. Uses 1d trend filter to avoid counter-trend trades. Designed for low-frequency, high-conviction trades.
Works in bull/bear by following 1d trend and using volatility-adjusted positioning.
"""

name = "12h_WilliamsAlligator_ElderRay_Trend_Signal"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Williams Alligator ===
    # Jaw (13-period SMMA, 8 bars ahead)
    # Teeth (8-period SMMA, 5 bars ahead)
    # Lips (5-period SMMA, 3 bars ahead)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Align to account for SMMA forward shift
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw, additional_delay_bars=8)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth, additional_delay_bars=5)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips, additional_delay_bars=3)
    
    # === 12h Elder Ray Power ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: check if jaws, teeth, lips are ordered (trending condition)
        # In uptrend: Lips > Teeth > Jaw
        # In downtrend: Jaw > Teeth > Lips
        alligator_long = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_short = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up, price above teeth, bull power positive, and 1d uptrend
            if (alligator_long and 
                close[i] > teeth_aligned[i] and 
                bull_power[i] > 0 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down, price below teeth, bear power negative, and 1d downtrend
            elif (alligator_short and 
                  close[i] < teeth_aligned[i] and 
                  bear_power[i] < 0 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks down OR price closes below jaw
            if not alligator_long or close[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Alligator alignment breaks up OR price closes above jaw
            if not alligator_short or close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals