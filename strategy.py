#!/usr/bin/env python3
# 4h Vortex Indicator + Volume Spike + Daily Trend
# Hypothesis: Vortex Indicator identifies trend direction (VI+ > VI- for uptrend, VI- > VI+ for downtrend).
# Combines with daily EMA50 trend filter and volume spikes for confirmation.
# Works in both bull and bear markets by following Vortex-defined momentum.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "4h_Vortex_Volume_DailyTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Vortex Indicator (14-period) ===
    tr1 = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First period TR
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # Sum over 14 periods
    tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    vm_plus_14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_14 / tr14
    vi_minus = vm_minus_14 / tr14
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: VI+ > VI- (uptrend) + volume spike + price above daily EMA50
            if (vi_plus[i] > vi_minus[i] and 
                vol_spike[i] and
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (downtrend) + volume spike + price below daily EMA50
            elif (vi_minus[i] > vi_plus[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend weakens (VI- crosses above VI+)
            if vi_minus[i] >= vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens (VI+ crosses above VI-)
            if vi_plus[i] >= vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals