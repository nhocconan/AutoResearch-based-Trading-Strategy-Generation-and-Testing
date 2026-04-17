#!/usr/bin/env python3
"""
1d CCI Trend Reversal with Weekly Trend Filter
Long: CCI(20) crosses above -100 + price above weekly EMA50
Short: CCI(20) crosses below +100 + price below weekly EMA50
Exit: Opposite CCI cross
Uses daily CCI for entry timing, weekly EMA50 for trend filter.
Designed to capture trend reversals in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # CCI(20) on daily
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need CCI calculation
    
    for i in range(start_idx, n):
        if (np.isnan(cci[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI crosses above -100 + price above weekly EMA50
            if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below +100 + price below weekly EMA50
            elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses below +100
            if cci[i] < 100 and cci[i-1] >= 100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses above -100
            if cci[i] > -100 and cci[i-1] <= -100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_CCI_TrendReversal_WeeklyEMA50"
timeframe = "1d"
leverage = 1.0