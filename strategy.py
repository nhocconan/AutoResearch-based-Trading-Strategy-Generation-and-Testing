#!/usr/bin/env python3
"""
6h_Aroon_Retracement_1dTrend
Hypothesis: Aroon measures trend strength and retracement depth. In trending markets (Aroon Up/Down > 70), 
pullbacks to Aroon oscillator zero line offer high-probability re-entry. Combined with 1d EMA50 trend filter,
this captures continuation moves in both bull (buy Aroon Up retracements) and bear (sell Aroon Down retracements).
Target: 15-25 trades/year per side. Uses Aroon(25) for smooth signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Aroon(25) - measures time since high/low
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low in lookback period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        # Calculate periods since high/low
        periods_since_high = period - 1 - np.where(high[i - period + 1:i + 1] == highest_high)[0][-1]
        periods_since_low = period - 1 - np.where(low[i - period + 1:i + 1] == lowest_low)[0][-1]
        
        aroon_up[i] = ((period - 1 - periods_since_high) / (period - 1)) * 100
        aroon_down[i] = ((period - 1 - periods_since_low) / (period - 1)) * 100
    
    # Aroon oscillator: Aroon Up - Aroon Down (positive = uptrend bias, negative = downtrend bias)
    aroon_osc = aroon_up - aroon_down
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Aroon calculation
    start_idx = period - 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(aroon_osc[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        aroon_up_val = aroon_up[i]
        aroon_down_val = aroon_down[i]
        aroon_osc_val = aroon_osc[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Aroon Up > 70 (strong uptrend) AND Aroon oscillator crosses above zero (retracement end)
            # AND price above 1d EMA50 (bullish regime)
            if aroon_up_val > 70 and aroon_osc_val > 0 and aroon_osc[i-1] <= 0 and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: Aroon Down > 70 (strong downtrend) AND Aroon oscillator crosses below zero (retracement end)
            # AND price below 1d EMA50 (bearish regime)
            elif aroon_down_val > 70 and aroon_osc_val < 0 and aroon_osc[i-1] >= 0 and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Aroon Down > 70 (downtrend taking over) OR Aroon oscillator < -20 (strong bearish bias)
            if aroon_down_val > 70 or aroon_osc_val < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Aroon Up > 70 (uptrend taking over) OR Aroon oscillator > 20 (strong bullish bias)
            if aroon_up_val > 70 or aroon_osc_val > 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Aroon_Retracement_1dTrend"
timeframe = "6h"
leverage = 1.0