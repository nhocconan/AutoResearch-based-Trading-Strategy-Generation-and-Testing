#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray power with 1d trend filter.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND close > 1d EMA50.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND close < 1d EMA50.
Exit when Alligator alignment reverses or price crosses 8-period EMA on 6h.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year per symbol.
Williams Alligator identifies trend phases via smoothed medians. Elder Ray measures bull/bear power
relative to EMA13. Combined with 1d EMA50 trend filter, this avoids counter-trend whipsaws in ranging markets.
"""

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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: 3 smoothed medians (Jaw=13, Teeth=8, Lips=5)
    # Smoothed with SMMA (smoothed moving average) = EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value: SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent: SMMA(t) = (SMMA(t-1)*(period-1) + price(t)) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 6h EMA8 for exit signal
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish alignment AND Bull Power > 0 AND close > 1d EMA50
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # Jaws < Teeth < Lips
                bull_power[i] > 0 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND Bear Power < 0 AND close < 1d EMA50
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Jaws > Teeth > Lips
                  bear_power[i] < 0 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator alignment reverses
            if position == 1 and not (jaw[i] < teeth[i] and teeth[i] < lips[i]):
                exit_signal = True
            elif position == -1 and not (jaw[i] > teeth[i] and teeth[i] > lips[i]):
                exit_signal = True
            
            # Secondary exit: price crosses 8-period EMA (fast reversal)
            if position == 1 and close[i] < ema8[i]:
                exit_signal = True
            elif position == -1 and close[i] > ema8[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Alligator_ElderRay_1dEMA50"
timeframe = "6h"
leverage = 1.0