#!/usr/bin/env python3
name = "6h_BullBear_Power_Zero_Cross"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # 1d EMA13 and EMA26 for Elder Ray calculation
    close_1d = df_1d['close']
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema26_1d = close_1d.ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate Bull Power and Bear Power (1d)
    bull_power_1d = high.max(axis=0) if hasattr(high, 'max') else np.maximum.reduce(high)  # This is wrong, need to fix
    # Actually, we need to compute per day: Bull Power = Daily High - EMA13
    # Since we have daily data in df_1d, we can compute:
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema26_1d
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6m EMA for trend context (optional filter)
    ema50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 26)  # Wait for EMA50 and EMA26
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema50_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero AND Bear Power below zero (bullish divergence)
            # Use previous value to detect cross
            if i > 0:
                bull_prev = bull_power_aligned[i-1]
                bear_prev = bear_power_aligned[i-1]
                bull_curr = bull_power_aligned[i]
                bear_curr = bear_power_aligned[i]
                
                # Bull Power crosses above zero while Bear Power is negative
                if bull_prev <= 0 and bull_curr > 0 and bear_curr < 0:
                    # Additional filter: price above 6h EMA50 for stronger trend
                    if close[i] > ema50_6h[i]:
                        signals[i] = 0.25
                        position = 1
                # Bear Power crosses below zero while Bull Power is positive
                elif bear_prev >= 0 and bear_curr < 0 and bull_curr > 0:
                    if close[i] < ema50_6h[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit: Bear Power crosses above zero OR Bull Power crosses below zero
            if i > 0:
                bull_prev = bull_power_aligned[i-1]
                bear_prev = bear_power_aligned[i-1]
                bull_curr = bull_power_aligned[i]
                bear_curr = bear_power_aligned[i]
                
                if bear_prev <= 0 and bear_curr > 0:  # Bear Power crosses above zero
                    signals[i] = 0.0
                    position = 0
                elif bull_prev >= 0 and bull_curr < 0:  # Bull Power crosses below zero
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power crosses above zero OR Bear Power crosses below zero
            if i > 0:
                bull_prev = bull_power_aligned[i-1]
                bear_prev = bear_power_aligned[i-1]
                bull_curr = bull_power_aligned[i]
                bear_curr = bear_power_aligned[i]
                
                if bull_prev <= 0 and bull_curr > 0:  # Bull Power crosses above zero
                    signals[i] = 0.0
                    position = 0
                elif bear_prev >= 0 and bear_curr < 0:  # Bear Power crosses below zero
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull Power/Bear Power) zero-cross system on 6h timeframe.
# Bull Power = Daily High - EMA13, Bear Power = Daily Low - EMA26.
# Long when Bull Power crosses above zero while Bear Power is negative (bullish divergence).
# Short when Bear Power crosses below zero while Bull Power is positive (bearish divergence).
# Uses 1d data for Elder Ray calculation to filter 6h noise, 6h EMA50 for trend alignment.
# Zero-cross signals reduce whipsaw vs. threshold-based systems.
# Works in bull markets (bull power dominance) and bear markets (bear power dominance).
# Discrete 0.25 position size limits risk. Target: 15-35 trades/year.