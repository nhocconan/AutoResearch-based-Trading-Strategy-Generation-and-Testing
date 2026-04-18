#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Momentum_v1
Strategy: 12h Camarilla pivot R1/S1 breakout with volume confirmation and momentum filter.
Long: Price breaks above R1 with volume > 1.5x average and momentum positive.
Short: Price breaks below S1 with volume > 1.5x average and momentum negative.
Momentum filter: 12-period RSI > 50 for long, < 50 for short.
Designed for 12h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via momentum filter and volume confirmation to avoid false breakouts.
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate 12-period RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12 = 100 - (100 / (1 + rs))
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_12)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Momentum filter
        mom_long = rsi_aligned[i] > 50
        mom_short = rsi_aligned[i] < 50
        
        # Breakout conditions
        breakout_long = high[i] > r1_aligned[i]
        breakout_short = low[i] < s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume + momentum
            if breakout_long and vol_confirm and mom_long:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume + momentum
            elif breakout_short and vol_confirm and mom_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below S1 or momentum shift
            if low[i] < s1_aligned[i] or not mom_long:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1 or momentum shift
            if high[i] > r1_aligned[i] or not mom_short:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Momentum_v1"
timeframe = "12h"
leverage = 1.0