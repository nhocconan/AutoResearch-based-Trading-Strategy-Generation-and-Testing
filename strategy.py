#!/usr/bin/env python3
"""
4h_1d_RVOL_Reversal_LongOnly
Hypothesis: Long-only strategy using 4-hour volume spike + price pullback to EMA21.
Enter long when: (1) volume > 2x 20-period average (RVOL > 2.0), (2) price pulls back to EMA21 within 0.5%,
(3) price > EMA50 (uptrend filter). Exit when price closes below EMA21 or RVOL drops below 1.2.
Designed for 4h timeframe to capture mean-reversion bounces in both bull and bear markets.
Volume spike indicates exhaustion; pullback to EMA21 offers favorable risk-reward.
EMA50 filter ensures we only trade in the direction of intermediate trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA21 and EMA50
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = np.where(vol_ma20 > 0, volume / vol_ma20, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(rvol[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: volume spike + pullback to EMA21 + above EMA50
            pullback_to_ema21 = abs(price - ema21[i]) / ema21[i] < 0.005  # within 0.5%
            if rvol[i] > 2.0 and pullback_to_ema21 and price > ema50[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit conditions: close below EMA21 or RVOL drops
            if price < ema21[i] or rvol[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "4h_1d_RVOL_Reversal_LongOnly"
timeframe = "4h"
leverage = 1.0