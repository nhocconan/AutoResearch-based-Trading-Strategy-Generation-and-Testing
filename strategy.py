# %pip install git+https://github.com/ta-lib/ta-lib-python.git

#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 4h Vortex indicator for trend direction + 1d volume spike filter.
Vortex captures trend strength and direction, effective in both bull and bear markets.
Volume > 2x average confirms trend strength. Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
Target: 20-50 trades/year (80-200 over 4 years). Includes ATR-based stoploss to limit drawdown.
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
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d volume average to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Vortex indicator on 4h data
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Vortex components
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Sum over 14 periods
    period = 14
    sum_tr = np.full(n, np.nan)
    sum_vm_plus = np.full(n, np.nan)
    sum_vm_minus = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr[i] = np.sum(tr[i-period+1:i+1])
        sum_vm_plus[i] = np.sum(vm_plus[i-period+1:i+1])
        sum_vm_minus[i] = np.sum(vm_minus[i-period+1:i+1])
    
    # VI+ and VI-
    vi_plus = np.full(n, np.nan)
    vi_minus = np.full(n, np.nan)
    for i in range(period, n):
        if sum_tr[i] > 0:
            vi_plus[i] = sum_vm_plus[i] / sum_tr[i]
            vi_minus[i] = sum_vm_minus[i] / sum_tr[i]
    
    # ATR for stoploss
    atr = np.full(n, np.nan)
    for i in range(period, n):
        if i == period:
            atr[i] = np.mean(tr[1:period+1])
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 14 for Vortex/ATR, 20 for 1d volume
    start_idx = max(period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(vi_plus[i]) or
            np.isnan(vi_minus[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        
        # Determine trend from Vortex: VI+ > VI- = bullish, VI- > VI+ = bearish
        bullish = vi_plus[i] > vi_minus[i]
        bearish = vi_minus[i] > vi_plus[i]
        
        # Volume confirmation: > 2x average daily volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: bullish Vortex + volume confirmation
            if bullish and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: bearish Vortex + volume confirmation
            elif bearish and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish Vortex or stoploss hit
            if bearish or price < (entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: bullish Vortex or stoploss hit
            if bullish or price > (entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
        
        # Track entry price for stoploss calculation
        if position != 0 and signals[i] != 0:
            if position == 1 and signals[i] == size:
                entry_price = price
            elif position == -1 and signals[i] == -size:
                entry_price = price
    
    return signals

name = "4h_Vortex_1dVolumeSpike"
timeframe = "4h"
leverage = 1.0