#/usr/bin/env python3
# 6H_Aroon_Oscillator_1dTrend_Filter
# Hypothesis: Uses Aroon Oscillator (25) on 6h to detect trend strength and direction,
# filtered by 1-day trend (close > EMA50 for long, close < EMA50 for short).
# Aroon Oscillator > 40 indicates strong uptrend, < -40 strong downtrend.
# This filters out weak trends and whipsaws, working in both bull and bear markets.
# Targets 12-30 trades per year on 6h timeframe with position size 0.25.

name = "6H_Aroon_Oscillator_1dTrend_Filter"
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
    
    # Get 1d data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Aroon Oscillator (25) on 6h
    period = 25
    
    def aroon_up(high, lookback):
        n = len(high)
        aroon_up = np.full(n, np.nan)
        for i in range(lookback-1, n):
            window_start = i - lookback + 1
            highest_idx = np.argmax(high[window_start:i+1]) + window_start
            periods_since_high = i - highest_idx
            aroon_up[i] = ((lookback - periods_since_high) / lookback) * 100
        return aroon_up
    
    def aroon_down(low, lookback):
        n = len(low)
        aroon_down = np.full(n, np.nan)
        for i in range(lookback-1, n):
            window_start = i - lookback + 1
            lowest_idx = np.argmin(low[window_start:i+1]) + window_start
            periods_since_low = i - lowest_idx
            aroon_down[i] = ((lookback - periods_since_low) / lookback) * 100
        return aroon_down
    
    aroon_up_val = aroon_up(high, period)
    aroon_down_val = aroon_down(low, period)
    aroon_osc = aroon_up_val - aroon_down_val  # -100 to +100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(aroon_osc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Aroon Oscillator > 40 (strong uptrend) in uptrend regime
            if (aroon_osc[i] > 40 and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: Aroon Oscillator < -40 (strong downtrend) in downtrend regime
            elif (aroon_osc[i] < -40 and price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Aroon Oscillator < 0 (trend weakening) or trend filter fails
            if (aroon_osc[i] < 0 or not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Aroon Oscillator > 0 (trend weakening) or trend filter fails
            if (aroon_osc[i] > 0 or not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals