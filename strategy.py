#!/usr/bin/env python3
"""
4h_1d_vortex_vortex_trend_v1
Hypothesis: 4-hour Vortex Indicator trend with 1-day volume confirmation and ATR volatility filter.
Vortex identifies trending markets (VI+ > VI-) and avoids ranging periods. Works in bull/bear by
filtering trades only when trend is strong. Uses volatility-adjusted position sizing to manage risk.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "4h_1d_vortex_vortex_trend_v1"
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
    
    # Get daily data for Vortex calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Vortex Indicator calculation (14-period)
    # True Range
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # VM+ and VM-
    vm_plus = np.abs(np.subtract(high_1d, np.roll(low_1d, 1)))
    vm_minus = np.abs(np.subtract(low_1d, np.roll(high_1d, 1)))
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Vortex trend: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend
    vortex_trend = vi_plus - vi_minus  # Positive = uptrend, Negative = downtrend
    
    # ATR for volatility filter (14-day ATR)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Align Vortex trend and ATR to 4h timeframe
    vortex_trend_aligned = align_htf_to_ltf(prices, df_1d, vortex_trend)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vortex_trend_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: strong uptrend (VI+ > VI-) with volume and volatility filter
        if (vortex_trend_aligned[i] > 0.1 and vol_confirm[i] and 
            atr_aligned[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: strong downtrend (VI- > VI+) with volume and volatility filter
        elif (vortex_trend_aligned[i] < -0.1 and vol_confirm[i] and 
              atr_aligned[i] > 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend reversal or volatility collapse
        elif position == 1 and (vortex_trend_aligned[i] < -0.05 or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vortex_trend_aligned[i] > 0.05 or not vol_confirm[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals