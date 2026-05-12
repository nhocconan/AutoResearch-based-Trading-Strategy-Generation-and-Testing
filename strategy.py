#!/usr/bin/env python3
# 12h_Vortex_1wTrend_Volume
# Hypothesis: Trade 12h breakouts of Vortex crossovers aligned with weekly trend and volume.
# Vortex identifies trend direction; weekly EMA50 filters long-term trend; volume confirms momentum.
# Designed for low frequency (15-30 trades/year) to survive both bull and bear markets.

name = "12h_Vortex_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Vortex Indicator (14-period) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +VM and -VM
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vi_plus[i]) or np.isnan(vi_minus[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Vortex crossover signals
        vi_cross_up = vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]
        vi_cross_down = vi_plus[i] < vi_minus[i] and vi_plus[i-1] >= vi_minus[i-1]
        
        # Volume filter: above 24-period average
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        vol_ok = not np.isnan(vol_ma_24[i]) and volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: VI+ crosses above VI-, uptrend, volume confirmation
            if vi_cross_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+, downtrend, volume confirmation
            elif vi_cross_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ or trend reversal
            if vi_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- or trend reversal
            if vi_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals