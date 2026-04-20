#!/usr/bin/env python3
# 4h_1D_ParabolicSAR_Reversal_V1
# Hypothesis: On 4h timeframe, trade reversals using Parabolic SAR with 1d trend filter.
# Parabolic SAR provides clear entry/exit signals. 1d EMA200 filters trend direction:
#   Long only when price > 1d EMA200 (bullish bias), short only when price < 1d EMA200 (bearish bias).
# Volume confirmation reduces false signals. Targets 20-40 trades/year by requiring trend alignment.
# Works in both bull and bear markets: in bull markets takes longs, in bear markets takes shorts.

name = "4h_1D_ParabolicSAR_Reversal_V1"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Parabolic SAR on 4h data
    # Initialize
    psar = np.zeros(n)
    bull = True  # Start assuming bullish
    af = 0.02    # Acceleration factor
    max_af = 0.2
    ep = high[0] if bull else low[0]  # Extreme point
    psar[0] = low[0] if bull else high[0]
    
    for i in range(1, n):
        # PSAR formula
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        
        # Handle reversals
        if bull:
            # Ensure PSAR doesn't exceed previous two lows
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
            # Check for bearish reversal
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep  # SAR becomes prior EP
                ep = low[i]   # Reset EP to current low
                af = 0.02     # Reset AF
            else:
                # Continue bullish
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            # Ensure PSAR doesn't fall below previous two highs
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
            # Check for bullish reversal
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep  # SAR becomes prior EP
                ep = high[i]  # Reset EP to current high
                af = 0.02     # Reset AF
            else:
                # Continue bearish
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema200_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for SAR reversal with trend and volume confirmation
            # Bullish SAR flip (PSAR moves below price) + uptrend filter + volume
            if (psar[i] < close[i] and psar[i-1] >= close[i-1] and  # Bullish flip
                close[i] > ema200_aligned[i] and                    # Above 1d EMA200
                volume[i] > 1.5 * volume_ma[i]):                    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Bearish SAR flip (PSAR moves above price) + downtrend filter + volume
            elif (psar[i] > close[i] and psar[i-1] <= close[i-1] and  # Bearish flip
                  close[i] < ema200_aligned[i] and                    # Below 1d EMA200
                  volume[i] > 1.5 * volume_ma[i]):                    # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: SAR flip to bearish or price drops below EMA200
            if psar[i] > close[i] or close[i] < ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: SAR flip to bullish or price rises above EMA200
            if psar[i] < close[i] or close[i] > ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals