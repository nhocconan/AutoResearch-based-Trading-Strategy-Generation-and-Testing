#!/usr/bin/env python3
"""
4h_WeeklyHighLow_Breakout_1dTrend_Volume
Hypothesis: Breakout above weekly high or below weekly low with 1d trend filter and volume confirmation.
Long when price breaks above weekly high in 1d uptrend with volume > 1.8x avg.
Short when price breaks below weekly low in 1d downtrend with volume > 1.8x avg.
Exit on price returning to weekly midpoint (average of weekly high/low) or trend reversal.
Weekly context provides stronger support/resistance than daily, reducing false breakouts.
Target: 15-25 trades/year to minimize fee decay while capturing major moves.
"""

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
    
    # Get weekly data for high/low levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly high and low
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate weekly midpoint for exit (average of weekly high/low)
    weekly_midpoint = (weekly_high + weekly_low) / 2
    
    # Align weekly and daily indicators to 4h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_weekly, weekly_midpoint)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h volume confirmation (20-period average)
    vol_ma_period = 20
    vol_ma_4h = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma_4h[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(1, 34, 20)  # Weekly needs 1 week, EMA(34), vol MA(20)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(weekly_midpoint_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h[i] if vol_ma_4h[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA34
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.8x average 4h volume
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price breaks above weekly high in uptrend with volume
            if uptrend and volume_confirmation and price > weekly_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low in downtrend with volume
            elif downtrend and volume_confirmation and price < weekly_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to weekly midpoint or trend reverses
            if price <= weekly_midpoint_aligned[i] or price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to weekly midpoint or trend reverses
            if price >= weekly_midpoint_aligned[i] or price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_WeeklyHighLow_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0