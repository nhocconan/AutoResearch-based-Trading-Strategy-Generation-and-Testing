#!/usr/bin/env python3
# 4h_1d_1w_camarilla_breakout_volume_trend_v1
# Hypothesis: 4-hour strategy combining daily Camarilla H3/L3 breakouts with weekly trend filter from SMA50/200 and volume confirmation.
# Weekly trend ensures directional bias aligned with higher timeframe, reducing false breakouts in sideways markets.
# Volume filter confirms breakout strength. Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in both bull and bear markets by following weekly trend direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (SMA50 > SMA200 = bullish)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMAs
    close_1w = df_1w['close'].values
    sma50_1w = np.full(len(close_1w), np.nan)
    sma200_1w = np.full(len(close_1w), np.nan)
    
    for i in range(50, len(close_1w)):
        sma50_1w[i] = np.mean(close_1w[i-50:i])
    for i in range(200, len(close_1w)):
        sma200_1w[i] = np.mean(close_1w[i-200:i])
    
    # Weekly trend: 1 = bullish (SMA50 > SMA200), -1 = bearish (SMA50 < SMA200), 0 = neutral
    weekly_trend = np.where(sma50_1w > sma200_1w, 1, np.where(sma50_1w < sma200_1w, -1, 0))
    
    # Align weekly trend to 4h timeframe
    weekly_trend_4h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels using PREVIOUS day's data (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no data yet)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_4h[i]) or np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average (strict filter for strength)
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Only take trades in direction of weekly trend
        weekly_bullish = weekly_trend_4h[i] == 1
        weekly_bearish = weekly_trend_4h[i] == -1
        
        # Entry conditions: Camarilla H3/L3 breakout with volume confirmation and weekly trend alignment
        long_breakout = (close[i] > h3_4h[i]) and volume_filter and weekly_bullish
        short_breakout = (close[i] < l3_4h[i]) and volume_filter and weekly_bearish
        
        # Exit conditions: touch opposite H3/L3 level or weekly trend reversal
        long_exit = (close[i] < l3_4h[i]) or (weekly_trend_4h[i] == -1)
        short_exit = (close[i] > h3_4h[i]) or (weekly_trend_4h[i] == 1)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0