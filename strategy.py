#!/usr/bin/env python3
"""
4H_Vortex_Vortex_Crossover_1dTrend_Filter_Volume
Hypothesis: Vortex indicator (VI+ and VI-) identifies trend direction with less whipsaw than moving averages. 
Combined with 1d trend filter (price vs 50 EMA) and volume confirmation (>2x average) to ensure trades align with higher timeframe momentum. 
Designed for 4h timeframe to capture strong trends with minimal false signals. 
Works in both bull and bear markets by following 1d trend direction, avoiding counter-trend trades.
Target: 20-40 trades/year (<160 total over 4 years) to minimize fee drag.
"""

name = "4H_Vortex_Vortex_Crossover_1dTrend_Filter_Volume"
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
    
    # Get 1d data for trend filter (50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Vortex Indicator (VI) on 4h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # VM+ and VM-
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and VI
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: VI+ crosses above VI- + above 1d EMA + volume spike
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: VI- crosses above VI+ + below 1d EMA + volume spike
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VI- crosses above VI+ or volume drops below average
            if (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VI+ crosses above VI- or volume drops below average
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals