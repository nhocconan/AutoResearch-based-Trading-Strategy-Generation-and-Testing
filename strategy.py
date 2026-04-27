#!/usr/bin/env python3
"""
6h_Pivot_Range_Breakout_1dTrend_Volume
Hypothesis: Combines daily pivot point ranges (PP, R1, S1) with breakout logic on 6h timeframe. Uses 1d EMA50 for trend filter and volume confirmation to avoid false breakouts. Targets 15-25 trades/year on 6h to minimize fee fade while capturing institutional moves in both bull and bear markets. Pivot ranges provide clear support/resistance levels that work in ranging markets, while breakouts with trend/volume filters capture trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d_prev) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, pivot calculation, volume MA
    start_idx = max(50, 30)  # EMA50, VolMA30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        pp_level = pp_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            if close[i] > r1_level and close[i] > ema_trend and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: break below S1 with downtrend and volume spike
            elif close[i] < s1_level and close[i] < ema_trend and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: break below PP or trend turns down
            if close[i] < pp_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above PP or trend turns up
            if close[i] > pp_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Pivot_Range_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0