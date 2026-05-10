#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Trend_Filter
Hypothesis: Price breaks the weekly Donchian channel (high/low of past 12 weeks) with trend filter from weekly EMA40 and volume confirmation.
Weekly Donchian captures longer-term structure, EMA40 filters for trend alignment, volume confirms breakout strength.
Works in bull/bear by only taking breakouts in direction of weekly trend. Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "1d_Weekly_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Donchian channel and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channel (12-week lookback)
    donchian_period = 12
    high_donchian = np.full(len(high_1w), np.nan)
    low_donchian = np.full(len(low_1w), np.nan)
    
    if len(high_1w) >= donchian_period:
        for i in range(donchian_period, len(high_1w)):
            high_donchian[i] = np.max(high_1w[i-donchian_period:i])
            low_donchian[i] = np.min(low_1w[i-donchian_period:i])
    
    # Weekly EMA40 for trend filter
    ema40_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 40:
        ema40_1w[39] = np.mean(close_1w[:40])
        alpha = 2 / (40 + 1)
        for i in range(40, len(close_1w)):
            ema40_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema40_1w[i-1]
    
    # Weekly volume SMA10 for volume confirmation
    vol_sma10_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 10:
        vol_sma10_1w[9] = np.mean(volume_1w[:10])
        for i in range(10, len(volume_1w)):
            vol_sma10_1w[i] = (vol_sma10_1w[i-1] * 9 + volume_1w[i]) / 10
    
    # Align 1w indicators to 1d
    high_donchian_aligned = align_htf_to_ltf(prices, df_1w, high_donchian)
    low_donchian_aligned = align_htf_to_ltf(prices, df_1w, low_donchian)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    vol_sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for EMA40
    
    for i in range(start_idx, n):
        if np.isnan(high_donchian_aligned[i]) or np.isnan(low_donchian_aligned[i]) or np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_sma10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x average 1w volume (scaled)
        # 1w = 5 x 1d bars, so scale weekly volume to daily equivalent
        vol_1w_scaled = vol_sma10_1w_aligned[i] / 5.0  # Average 1d-equivalent volume from 1w data
        volume_confirm = volume[i] > 1.3 * vol_1w_scaled
        
        # Trend and price relative to weekly Donchian levels
        is_uptrend = close[i] > ema40_1w_aligned[i]
        is_downtrend = close[i] < ema40_1w_aligned[i]
        price_above_donchian_high = close[i] > high_donchian_aligned[i]
        price_below_donchian_low = close[i] < low_donchian_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, in uptrend, with volume
            if price_above_donchian_high and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low, in downtrend, with volume
            elif price_below_donchian_low and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly Donchian high or trend turns down
            if not price_above_donchian_high or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly Donchian low or trend turns up
            if not price_below_donchian_low or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals