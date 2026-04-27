#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_1wTrend_CloseFilter
Hypothesis: Weekly Donchian(20) breakout with 1-week trend filter on daily chart.
Enter long when price breaks above weekly Donchian high and weekly trend is up.
Enter short when price breaks below weekly Donchian low and weekly trend is down.
Exit when price closes back inside the weekly Donchian channel.
Designed to work in both bull and bear markets by following weekly trend with volatility breakout.
Target: 10-25 trades/year to minimize fee drag.
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
    
    # Get weekly data for Donchian calculation and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    period = 20
    for i in range(period - 1, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i - period + 1:i + 1])
        donchian_low[i] = np.min(low_1w[i - period + 1:i + 1])
    
    # Weekly trend: price above/below 21-period EMA on weekly close
    ema_period = 21
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i - 1] * (1 - multiplier))
    
    # Align weekly indicators to daily timeframe
    dh_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(period, ema_period)
    
    for i in range(start_idx, n):
        if (np.isnan(dh_aligned[i]) or
            np.isnan(dl_aligned[i]) or
            np.isnan(ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Weekly trend filter
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high in uptrend
            if uptrend and price > dh_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low in downtrend
            elif downtrend and price < dl_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes back inside weekly Donchian channel
            if price < dh_aligned[i] and price > dl_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price closes back inside weekly Donchian channel
            if price < dh_aligned[i] and price > dl_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_WeeklyDonchianBreakout_1wTrend_CloseFilter"
timeframe = "1d"
leverage = 1.0