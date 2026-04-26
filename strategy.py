#!/usr/bin/env python3
"""
6h_WeeklyPivot_Camarilla_Breakout_v1
Hypothesis: 6h Camarilla breakout with weekly pivot direction filter and volume spike confirmation.
- Long when price breaks above weekly R3 AND weekly close > weekly open (bullish weekly candle) AND volume > 2.0 * volume_ma(20)
- Short when price breaks below weekly S3 AND weekly close < weekly open (bearish weekly candle) AND volume > 2.0 * volume_ma(20)
- Uses weekly Camarilla levels (R3/S3) from 1w chart for structure-based breakouts
- Weekly candle direction filter ensures trading with higher timeframe momentum
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 12-37 trades/year on 6h) to minimize fee drag
- Exit on opposite weekly Camarilla level (S3 for longs, R3 for shorts) or weekly candle reversal
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla levels and candle direction
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels (R3, S3, R4, S4)
    # Weekly pivot point
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_range = df_1w['high'] - df_1w['low']
    # Camarilla levels
    r3 = weekly_pivot + weekly_range * 1.1 / 4
    s3 = weekly_pivot - weekly_range * 1.1 / 4
    r4 = weekly_pivot + weekly_range * 1.1 / 2
    s4 = weekly_pivot - weekly_range * 1.1 / 2
    # Weekly candle direction: 1 = bullish (close > open), -1 = bearish (close < open), 0 = doji/neutral
    weekly_bullish = df_1w['close'] > df_1w['open']
    weekly_bearish = df_1w['close'] < df_1w['open']
    weekly_direction = np.where(weekly_bullish, 1, np.where(weekly_bearish, -1, 0))
    
    # Align weekly data to 6h timeframe (completed weekly candle only)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    weekly_direction_aligned = align_htf_to_ltf(prices, df_1w, weekly_direction.values)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 1 for weekly data, 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(weekly_direction_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly Camarilla breakout conditions with candle direction and volume spike filter
        if position == 0:
            # Long: Price breaks above weekly R3 AND weekly bullish candle AND volume spike
            if close[i] > r3_aligned[i] and weekly_direction_aligned[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S3 AND weekly bearish candle AND volume spike
            elif close[i] < s3_aligned[i] and weekly_direction_aligned[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below weekly S3 OR weekly candle turns bearish
            if close[i] < s3_aligned[i] or weekly_direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above weekly R3 OR weekly candle turns bullish
            if close[i] > r3_aligned[i] or weekly_direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Camarilla_Breakout_v1"
timeframe = "6h"
leverage = 1.0