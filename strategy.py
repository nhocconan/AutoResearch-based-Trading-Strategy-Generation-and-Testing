#!/usr/bin/env python3
# 12h_Vortex_Trend_Volume_Confirmation
# Hypothesis: Vortex indicator identifies trend direction on 12h timeframe; combined with weekly trend filter and volume spike.
# Vortex > 1 indicates bullish trend, < 1 indicates bearish trend. Weekly trend ensures alignment with higher timeframe momentum.
# Volume confirmation filters out false breakouts. Designed for low trade frequency (15-25/year) to minimize fee drag.
# Works in both bull and bear markets by following the dominant trend on multiple timeframes.

name = "12h_Vortex_Trend_Volume_Confirmation"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Vortex indicator (14-period)
    # VM+ = |High - Prior Low|
    # VM- = |Low - Prior High|
    # Sum of VM+ and VM- over 14 periods
    # VI+ = Sum(VM+) / Sum(TR)
    # VI- = Sum(VM-) / Sum(TR)
    # Where TR = True Range
    
    # Calculate true range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate VM+ and VM-
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    # First values are invalid due to roll
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    # Sum over 14 periods
    sum_vm_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Vortex indicators
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # Align weekly EMA to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (24-period MA on 12h chart ≈ 12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Vortex (14), weekly EMA (34), volume MA (24)
    start_idx = max(14, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Vortex trend strength
        bullish_vortex = vi_plus[i] > vi_minus[i] and vi_plus[i] > 1.0
        bearish_vortex = vi_minus[i] > vi_plus[i] and vi_minus[i] > 1.0
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: bullish Vortex + weekly uptrend + volume spike
            if bullish_vortex and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Vortex + weekly downtrend + volume spike
            elif bearish_vortex and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Vortex turns bearish or weekly trend turns down
            if not bullish_vortex or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Vortex turns bullish or weekly trend turns up
            if not bearish_vortex or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals