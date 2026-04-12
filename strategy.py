#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_elder_ray_power_v1
# Uses daily Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 6h price action.
# Goes long when Bull Power > 0 and 6h close > prior 6h close (momentum).
# Goes short when Bear Power > 0 and 6h close < prior 6h close.
# Uses 6h ATR(14) > 0 to avoid dead markets.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets (strong bull power + upward momentum) and bear markets (strong bear power + downward momentum).

name = "6h_1d_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on daily closes
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = df_1d['high'].values - ema13  # High - EMA13
    bear_power = ema13 - df_1d['low'].values   # EMA13 - Low
    
    # Align to 6h timeframe (daily values update after daily bar closes)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h momentum: close > prior close
    mom = np.zeros(n)
    mom[1:] = close[1:] > close[:-1]
    
    # 6h volatility filter: ATR(14) > 0 (avoid dead markets)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(mom[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require volatility filter
        if not vol_filter[i]:
            # Hold current position if no volatility
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: Bull Power > 0 and upward momentum
        if bull_power_aligned[i] > 0 and mom[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: Bear Power > 0 and downward momentum
        elif bear_power_aligned[i] > 0 and not mom[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite power > 0
        elif bear_power_aligned[i] > 0 and position == 1:
            position = 0
            signals[i] = 0.0
        elif bull_power_aligned[i] > 0 and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals