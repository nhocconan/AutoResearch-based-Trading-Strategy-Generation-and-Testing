#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dVolumeFilter
Hypothesis: Use weekly pivot points (from prior week) for directional bias, combined with 1d volume confirmation. 
Go long when price > weekly pivot AND 1d volume > 1.5x 20-day average, short when price < weekly pivot AND volume confirmation.
Weekly pivots provide structural support/resistance that works in both trending and ranging markets.
Volume confirmation ensures trades occur with participation, reducing false breakouts.
Target: 20-40 trades/year by requiring both pivot alignment and volume spike.
Works in bull markets (buy dips above pivot) and bear (sell rallies below pivot).
"""

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
    
    # Get weekly data for pivot points (prior week's data)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    # We'll use the pivot point itself as the key level
    typical_price_weekly = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3.0
    pivot_weekly = typical_price_weekly.values
    
    # Align weekly pivot to 6h timeframe (will use prior week's pivot)
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume 20-period moving average
    vol_ma_period = 20
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    
    if len(volume_1d) >= vol_ma_period:
        for i in range(vol_ma_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - vol_ma_period:i])
    
    # Align 1d volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one bar of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_weekly_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation: current 6h volume > 1.5x 1d volume MA
        # Need to convert 6h volume to daily equivalent for comparison
        # Approximate: 4x 6h volume ≈ 1 day volume (since 4*6h = 24h)
        vol_6h_equiv = volume[i] * 4.0
        vol_confirm = vol_6h_equiv > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0 and in_session:
            # Long: price above weekly pivot + volume confirmation
            if close[i] > pivot_weekly_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot + volume confirmation
            elif close[i] < pivot_weekly_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot
            if close[i] < pivot_weekly_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot
            if close[i] > pivot_weekly_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Direction_1dVolumeFilter"
timeframe = "6h"
leverage = 1.0