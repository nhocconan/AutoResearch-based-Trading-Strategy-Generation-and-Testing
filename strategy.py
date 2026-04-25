#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume confirmation.
- Trend filter: price > 12h EMA50 = bullish, price < 12h EMA50 = bearish.
- In bullish 12h trend: buy breakouts above R1, sell breakdowns below S1.
- In bearish 12h trend: sell breakdowns below S1, buy breakouts above R1 (continuation logic).
- Volume confirmation: require volume > 2.0x 20-period average to avoid false breakouts.
- Exit on trend reversal or mean reversion to pivot.
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: 12h trend filter captures major moves, volume filter reduces noise.
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
    
    # Get 12h data for HTF trend filter and Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar's OHLC)
    prev_close = np.roll(df_12h['close'].values, 1)
    prev_high = np.roll(df_12h['high'].values, 1)
    prev_low = np.roll(df_12h['low'].values, 1)
    prev_close[0] = df_12h['close'].values[0]
    prev_high[0] = df_12h['high'].values[0]
    prev_low[0] = df_12h['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend using EMA50
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Breakout logic: trade in direction of 12h trend with volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_12h_bullish and volume_spike[i]
            short_setup = (close[i] < s1_aligned[i]) and htf_12h_bearish and volume_spike[i]
            
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
            # Exit on trend reversal or mean reversion to pivot
            exit_signal = (not htf_12h_bullish) or (close[i] < pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to pivot
            exit_signal = htf_12h_bullish or (close[i] > pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0