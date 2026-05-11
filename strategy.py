#!/usr/bin/env python3
"""
4h_Vortex_Volume_Spike_1dTrend
Hypothesis: Vortex indicator (VI+ > VI-) signals trend direction, filtered by 1d EMA50 trend and volume spike (1.5x median). 
Works in bull (VI+ > VI- + uptrend) and bear (VI- > VI+ + downtrend). 
Fewer trades via strict confluence: trend + volume + Vortex crossover. Target: 20-40 trades/year.
"""

name = "4h_Vortex_Volume_Spike_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Vortex Indicator (14-period) on 4h ---
    # True Range
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Vortex movement
    vm_plus = np.abs(high_4h - np.roll(low_4h, 1))
    vm_minus = np.abs(low_4h - np.roll(high_4h, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Sum over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and Vortex
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Simple exit: reverse signal or opposite Vortex crossover
                if position == 1 and vi_minus[i] > vi_plus[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and vi_plus[i] > vi_minus[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if vi_plus[i] > vi_minus[i] and trend_up and vol_ok:
                # Long: VI+ > VI- + 1d uptrend + volume spike
                signals[i] = 0.20
                position = 1
                entry_price = close_4h[i]
            elif vi_minus[i] > vi_plus[i] and trend_down and vol_ok:
                # Short: VI- > VI+ + 1d downtrend + volume spike
                signals[i] = -0.20
                position = -1
                entry_price = close_4h[i]
        else:
            # Exit on Vortex crossover in opposite direction
            if position == 1:
                if vi_minus[i] > vi_plus[i]:  # Vortex signals bearish
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                if vi_plus[i] > vi_minus[i]:  # Vortex signals bullish
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals