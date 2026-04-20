#!/usr/bin/env python3
# 12h_Vortex_Trend_Signal_With_Volume_Filter
# Hypothesis: Vortex Indicator captures trend direction (VI+ > VI- for uptrend, VI- > VI+ for downtrend).
# Combining with 1d trend filter (price vs 50 EMA) and volume confirmation creates a robust trend-following system.
# Works in bull markets by capturing uptrends and in bear markets by capturing downtrends.
# Volume filter ensures trades occur only with institutional participation, reducing false signals.
# Designed for 12h timeframe to limit trades to 50-150 over 4 years, minimizing fee drag.

name = "12h_Vortex_Trend_Signal_With_Volume_Filter"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Vortex Indicator components (VI+ and VI-)
    # VM+ = |high - low_prev|, VM- = |low - high_prev|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Sum over 14 periods
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.8x 30-period average (higher threshold to reduce trades)
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (uptrend) + price above 1d EMA50 + volume confirmation
            if vi_plus[i] > vi_minus[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (downtrend) + price below 1d EMA50 + volume confirmation
            elif vi_minus[i] > vi_plus[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend reverses (VI- > VI+)
            if vi_minus[i] > vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend reverses (VI+ > VI-)
            if vi_plus[i] > vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals