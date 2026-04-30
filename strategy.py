#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w trend filter.
# Uses 1w EMA50 for higher timeframe trend alignment (bull/bear regime).
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via convergence/divergence.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# Long when: price > Alligator Jaw, Bull Power > 0, Bear Power < 0, and price > 1w EMA50.
# Short when: price < Alligator Jaw, Bull Power < 0, Bear Power > 0, and price < 1w EMA50.
# Exit when Alligator lines re-converge (|Jaw-Teeth| < 0.1*ATR) or price crosses Jaw.
# Targets 12-37 trades/year on 6h with strong trend filtering to avoid whipsaws.
# Works in bull markets via Alligator alignment and in bear markets via inverse signals.

name = "6h_WilliamsAlligator_ElderRay_1wEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for exit condition
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator (SMMA = Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Calculate Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Alligator convergence check (for exit)
        jaw_teeth_dist = abs(curr_jaw - curr_teeth)
        alligator_converged = jaw_teeth_dist < (0.1 * curr_atr)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Jaw, Bull Power > 0, Bear Power < 0, price > 1w EMA50
            if (curr_close > curr_jaw and
                curr_bull > 0 and
                curr_bear < 0 and
                curr_close > curr_ema_50_1w):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw, Bull Power < 0, Bear Power > 0, price < 1w EMA50
            elif (curr_close < curr_jaw and
                  curr_bull < 0 and
                  curr_bear > 0 and
                  curr_close < curr_ema_50_1w):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Alligator re-converged OR price crosses below Jaw
            if alligator_converged or curr_close < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator re-converged OR price crosses above Jaw
            if alligator_converged or curr_close > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals