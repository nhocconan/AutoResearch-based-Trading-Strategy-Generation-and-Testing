#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation.
- Trend filter: price > 1d EMA34 = bullish, price < 1d EMA34 = bearish.
- In bullish 1d trend: buy breakouts above R1, sell breakdowns below S1.
- In bearish 1d trend: sell breakdowns below S1, buy breakouts above R1 (continuation logic).
- Volume confirmation: require volume > 2.0x 20-period average to avoid false breakouts.
- Exit on trend reversal or mean reversion to prior 12h Camarilla pivot (dynamic per 12h bar).
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in both bull and bear: 1d trend filter captures major moves, volume filter reduces noise, 12h pivot exit improves win rate.
- Primary timeframe: 12h, HTF: 1d for trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Camarilla pivot levels (dynamic per 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels using previous 12h bar's OHLC
    prev_close_12h = np.roll(df_12h['close'].values, 1)
    prev_high_12h = np.roll(df_12h['high'].values, 1)
    prev_low_12h = np.roll(df_12h['low'].values, 1)
    prev_close_12h[0] = df_12h['close'].values[0]
    prev_high_12h[0] = df_12h['high'].values[0]
    prev_low_12h[0] = df_12h['low'].values[0]
    
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    range_12h = prev_high_12h - prev_low_12h
    
    # Camarilla levels for 12h
    r1_12h = pivot_12h + (range_12h * 1.1 / 12)
    s1_12h = pivot_12h - (range_12h * 1.1 / 12)
    
    # Align 12h Camarilla levels to 12h timeframe (already aligned, but use helper for safety)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or
            np.isnan(pivot_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA34
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Breakout logic: trade in direction of 1d trend with volume confirmation
            long_setup = (close[i] > r1_12h_aligned[i]) and htf_1d_bullish and volume_spike[i]
            short_setup = (close[i] < s1_12h_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or mean reversion to 12h pivot
            exit_signal = (not htf_1d_bullish) or (close[i] < pivot_12h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to 12h pivot
            exit_signal = htf_1d_bullish or (close[i] > pivot_12h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0