#!/usr/bin/env python3
name = "6h_PivotPoint_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and pivot points
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Standard Pivot Points from previous day
    # Pivot Point = (High + Low + Close) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align pivot levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # Price range filter: avoid choppy markets (ATR-based)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0] if len(tr) > 0 else 0.0], tr]) if len(tr) > 0 else np.array([0.0])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Normalize ATR by price to get percentage
    atr_pct = atr / close
    # Only trade when volatility is moderate (not too high, not too low)
    vol_regime = (atr_pct > 0.008) & (atr_pct < 0.04)  # 0.8% to 4% ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_filter[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bounce from S1 + above 1d EMA50 + volume filter + vol regime
            if low[i] <= s1_1d_aligned[i] and close[i] > s1_1d_aligned[i] and close[i] > ema_50_1d_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: rejection at R1 + below 1d EMA50 + volume filter + vol regime
            elif high[i] >= r1_1d_aligned[i] and close[i] < r1_1d_aligned[i] and close[i] < ema_50_1d_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S1 or below 1d EMA50
            if low[i] < s1_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R1 or above 1d EMA50
            if high[i] > r1_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals