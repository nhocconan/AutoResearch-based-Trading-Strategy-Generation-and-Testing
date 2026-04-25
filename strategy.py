#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA50_VolumeConfirm_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA50 trend filter and volume spike confirmation.
- In trending markets (price > 1d EMA50): buy breakouts above R1, sell breakdowns below S1.
- In ranging markets (price near 1d EMA50): fade extremes at R1/S1 with mean reversion.
- Volume confirmation: require volume > 1.8x 20-period average to reduce false signals.
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: trend filter adapts to market regime, volume filters noise.
- Tighter volume threshold (1.8x vs 1.5x) and longer EMA (50 vs 34) to reduce trade frequency and improve Sharpe.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike confirmation: volume > 1.8x 20-period average (tighter than 1.5x)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Determine if we are in trending or ranging market based on distance from EMA
        ema_distance = abs(close[i] - ema_50_1d_aligned[i]) / ema_50_1d_aligned[i]
        trending_market = ema_distance > 0.025  # Slightly wider band (2.5%) to reduce whipsaws
        ranging_market = ema_distance <= 0.025
        
        if position == 0:
            if trending_market:
                # Trending market: trade breakout continuation
                long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_spike[i]
                short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_spike[i]
            else:
                # Ranging market: trade mean reversion at extremes
                long_setup = (close[i] < s1_aligned[i]) and (close[i] > s3_aligned[i]) and volume_spike[i]  # Oversold bounce
                short_setup = (close[i] > r1_aligned[i]) and (close[i] < r3_aligned[i]) and volume_spike[i]  # Overbought rejection
            
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
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of S1
                exit_signal = (not htf_1d_bullish) or (close[i] < s1_aligned[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of R1
                exit_signal = (close[i] > pivot_aligned[i]) or (close[i] > r1_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of R1
                exit_signal = htf_1d_bullish or (close[i] > r1_aligned[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of S1
                exit_signal = (close[i] < pivot_aligned[i]) or (close[i] < s1_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0