#!/usr/bin/env python3
"""
4h_Vortex_Trend_Filtered_By_12h_EMA
Hypothesis: Vortex Indicator (VI) identifies trend direction (VI+ > VI- for uptrend). 
Filtered by 12h EMA50 trend to avoid counter-trend trades. 
Entry: VI+ crosses above VI- AND price > 12h EMA50 (long); VI- crosses above VI+ AND price < 12h EMA50 (short).
Exit: Opposite Vortex cross. 
Position size: 0.25. 
Designed for 4h timeframe with 12h trend filter to work in both bull (follow uptrend) and bear (follow downtrend) markets.
Low trade frequency expected due to dual confirmation.
"""

name = "4h_Vortex_Trend_Filtered_By_12h_EMA"
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
    
    # Vortex Indicator (VI) on 4h - using 14 periods
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # Handle first element
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[1] if n > 1 else 0  # handle first element
    
    # Sum over 14 periods
    n_periods = 14
    vi_plus = pd.Series(vm_plus).rolling(window=n_periods, min_periods=n_periods).sum().values / \
              pd.Series(tr).rolling(window=n_periods, min_periods=n_periods).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=n_periods, min_periods=n_periods).sum().values / \
               pd.Series(tr).rolling(window=n_periods, min_periods=n_periods).sum().values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(n_periods, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: VI+ crosses above VI- AND price > 12h EMA50 (uptrend)
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ AND price < 12h EMA50 (downtrend)
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend change to down)
            if vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend change to up)
            if vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals