#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceAction_1dTrend_Volume
Hypothesis: Weekly pivot levels act as strong support/resistance. Price action rejection (pin bar) at these levels combined with 1-day trend and volume spike provides high-probability entries in both bull and bear markets. Weekly timeframe reduces noise, and tight entry conditions (pin bar + volume + trend) limit trades to avoid fee drag. Target: 15-25 trades/year per symbol.
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Pin bar detection (rejection candle)
        # Bullish pin: long lower shadow, small body, close near high
        body_size = abs(close[i] - prices['open'].iloc[i])
        lower_shadow = prices['open'].iloc[i] - low[i] if close[i] >= prices['open'].iloc[i] else close[i] - low[i]
        upper_shadow = high[i] - close[i] if close[i] >= prices['open'].iloc[i] else high[i] - prices['open'].iloc[i]
        
        # Avoid division by zero
        if body_size < 0.001:
            is_bullish_pin = False
            is_bearish_pin = False
        else:
            # Bullish pin: lower shadow >= 2x body, upper shadow <= 0.3x body
            is_bullish_pin = (lower_shadow >= 2 * body_size) and (upper_shadow <= 0.3 * body_size)
            # Bearish pin: upper shadow >= 2x body, lower shadow <= 0.3x body
            is_bearish_pin = (upper_shadow >= 2 * body_size) and (lower_shadow <= 0.3 * body_size)
        
        # Trend filter from daily EMA20
        uptrend = close[i] > ema_20_1d_aligned[i]
        downtrend = close[i] < ema_20_1d_aligned[i]
        
        # Proximity to weekly pivot levels (within 0.5% of level)
        proximity_threshold = 0.005  # 0.5%
        near_r1 = abs(close[i] - r1_aligned[i]) / r1_aligned[i] < proximity_threshold
        near_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < proximity_threshold
        near_r2 = abs(close[i] - r2_aligned[i]) / r2_aligned[i] < proximity_threshold
        near_s2 = abs(close[i] - s2_aligned[i]) / s2_aligned[i] < proximity_threshold
        near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < proximity_threshold
        near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < proximity_threshold
        
        # Entry conditions: pin bar at pivot level + volume spike + trend alignment
        long_entry = (is_bullish_pin and (near_s1 or near_s2 or near_s3)) and volume_spike[i] and uptrend
        short_entry = (is_bearish_pin and (near_r1 or near_r2 or near_r3)) and volume_spike[i] and downtrend
        
        # Exit on opposite signal or strong reversal
        long_exit = is_bearish_pin and (near_r1 or near_r2 or near_r3) and volume_spike[i]
        short_exit = is_bullish_pin and (near_s1 or near_s2 or near_s3) and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_PriceAction_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0