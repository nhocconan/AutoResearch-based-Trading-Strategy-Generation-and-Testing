#!/usr/bin/env python3
name = "1h_SMC_Liquidity_Sweep_1dTrend"
timeframe = "1h"
leverage = 1.0

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
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # === 1d Data for Trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Swing Points for Liquidity Detection ===
    # Find swing highs and lows using 3-bar lookback/forward
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(2, n - 2):
        # Swing high: higher than 2 bars before and after
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = True
        # Swing low: lower than 2 bars before and after
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = True
    
    # === Liquidity Sweep Detection ===
    # Bullish sweep: price makes new low but closes above swing low
    bullish_sweep = np.zeros(n, dtype=bool)
    bearish_sweep = np.zeros(n, dtype=bool)
    
    for i in range(3, n):
        # Check for bullish sweep: new low that closes above prior swing low
        if low[i] < low[i-1]:  # Made new low
            # Look back for any swing low that was breached
            for j in range(max(0, i-20), i):
                if swing_low[j] and low[i] < low[j] and close[i] > low[j]:
                    bullish_sweep[i] = True
                    break
        
        # Check for bearish sweep: new high that closes below prior swing high
        if high[i] > high[i-1]:  # Made new high
            # Look back for any swing high that was breached
            for j in range(max(0, i-20), i):
                if swing_high[j] and high[i] > high[j] and close[i] < high[j]:
                    bearish_sweep[i] = True
                    break
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Minimum for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish liquidity sweep + price above 1d EMA200 (uptrend)
            if bullish_sweep[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Bearish liquidity sweep + price below 1d EMA200 (downtrend)
            elif bearish_sweep[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Bearish sweep or trend breaks
            if bearish_sweep[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Bullish sweep or trend breaks
            if bullish_sweep[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals