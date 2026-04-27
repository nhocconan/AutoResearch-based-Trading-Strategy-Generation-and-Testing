#!/usr/bin/env python3
"""
#100778 - 1d_Technical_Trend_WeeklyFilter
Hypothesis: Daily trend following with weekly trend filter and volume confirmation.
Targets 10-25 trades/year to minimize fee drift. Works in bull (trend continuation) and bear (trend reversals).
Uses 1d primary timeframe with 1w EMA50 trend filter and ATR-based position sizing.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily ATR(14) for position sizing and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: close above weekly EMA50 and above daily open
        if (close[i] > ema50_1w_aligned[i] and 
            close[i] > prices['open'].iloc[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: close below weekly EMA50 and below daily open
        elif (close[i] < ema50_1w_aligned[i] and 
              close[i] < prices['open'].iloc[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses back over weekly EMA50
        elif position == 1 and close[i] < ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Technical_Trend_WeeklyFilter"
timeframe = "1d"
leverage = 1.0