#!/usr/bin/env python3
"""
6h_1D_Pivot_R1S1_Breakout_Volume_Imbalance_V1
Hypothesis: Use daily pivot R1/S1 for directional bias with 6H entry, combining volume confirmation and volume imbalance detection (current volume > 1.5x average AND buy/sell volume ratio > 1.2 or < 0.8) to filter false breakouts. Trade only during active session (08-20 UTC). Fixed position size 0.25. Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drift. Works in bull/bear via volume/imbalance filters and session timing.
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
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for standard pivot calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Standard pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align daily pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for volume averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Volume imbalance: buy/sell volume ratio
        sell_volume = volume[i] - taker_buy_volume[i]
        if sell_volume > 0:
            vol_ratio = taker_buy_volume[i] / sell_volume
        else:
            vol_ratio = 2.0  # assume strong buying if no sell volume
        
        # Imbalance confirmation: strong buying (>1.2) or strong selling (<0.8)
        imbalance_confirm = (vol_ratio > 1.2) or (vol_ratio < 0.8)
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and imbalance confirmation during session
            if close[i] > r1_aligned[i] and vol_confirm and imbalance_confirm and (vol_ratio > 1.2) and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and imbalance confirmation during session
            elif close[i] < s1_aligned[i] and vol_confirm and imbalance_confirm and (vol_ratio < 0.8) and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or volume/imbalance fails or outside session
            if close[i] < r1_aligned[i] or not vol_confirm or not imbalance_confirm or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or volume/imbalance fails or outside session
            if close[i] > s1_aligned[i] or not vol_confirm or not imbalance_confirm or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_Pivot_R1S1_Breakout_Volume_Imbalance_V1"
timeframe = "6h"
leverage = 1.0