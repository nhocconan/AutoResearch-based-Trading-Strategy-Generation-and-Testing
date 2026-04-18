#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Weekly_Filter
1d strategy using KAMA trend direction filtered by weekly trend and volume confirmation.
- KAMA direction: rising when price > KAMA, falling when price < KAMA
- Weekly filter: only take long when weekly KAMA is rising, short when falling
- Volume confirmation: volume > 1.5x 20-day average
- Position sizing: 0.25 long/short, 0.0 flat
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in bull markets (trend following) and bear markets (avoids false signals via weekly filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |change| / sum(|changes|) over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(change)
    volatility[1:] = np.abs(np.diff(close))
    
    # Avoid division by zero
    er = np.zeros_like(change, dtype=float)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 when price > KAMA, -1 when price < KAMA
    kama_direction = np.where(close > kama, 1, -1)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly KAMA
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_1w = np.zeros_like(change_1w)
    volatility_1w[1:] = np.abs(np.diff(close_1w))
    er_1w = np.zeros_like(change_1w, dtype=float)
    mask_1w = volatility_1w != 0
    er_1w[mask_1w] = change_1w[mask_1w] / volatility_1w[mask_1w]
    sc_1w = (er_1w * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_direction = np.where(close_1w > kama_1w, 1, -1)
    
    # Align weekly KAMA direction to daily
    kama_1w_dir_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_direction)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Daily volume average (20-period)
    vol_ma_20 = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1w_dir_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_up = kama_1w_dir_aligned[i] == 1
        weekly_down = kama_1w_dir_aligned[i] == -1
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # KAMA direction signal
        kama_up = kama_direction[i] == 1
        kama_down = kama_direction[i] == -1
        
        if position == 0:
            # Long: weekly up + volume + KAMA up
            if weekly_up and vol_confirm and kama_up:
                signals[i] = 0.25
                position = 1
            # Short: weekly down + volume + KAMA down
            elif weekly_down and vol_confirm and kama_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly down or KAMA down
            if not weekly_up or not kama_up:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly up or KAMA up
            if not weekly_down or not kama_down:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Weekly_Filter"
timeframe = "1d"
leverage = 1.0