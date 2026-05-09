#!/usr/bin/env python3
"""
12h Weekly Trend + Daily Breakout with Volume Confirmation
Hypothesis: In BTC/ETH, weekly trend direction (above/below 200 EMA) combined with 
daily breakout above/below Donchian channel (20) provides strong directional edge.
Volume surge confirms institutional participation. This should work in both bull 
(bullish breakouts in uptrend) and bear (bearish breakdowns in downtrend) markets.
Target: 15-25 trades/year to avoid fee drag.
"""

name = "12h_WeeklyTrend_DailyBreakout_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly EMA200 for trend filter
    ema200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema200_1w[i] = (close_1w[i] * 2 + ema200_1w[i-1] * 198) / 200
    
    # Align weekly EMA200 to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily Donchian channels (20-period)
    donchian_high_1d = np.full_like(close_1d, np.nan)
    donchian_low_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if i >= 19:
            donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
            donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
        else:
            donchian_high_1d[i] = np.max(high_1d[0:i+1]) if i >= 0 else np.nan
            donchian_low_1d[i] = np.min(low_1d[0:i+1]) if i >= 0 else np.nan
    
    # Align Donchian levels to 12h timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Daily volume filter: current volume vs 20-period average
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma_1d[i] = (vol_ma_1d[i-1] * 19 + volume_1d[i]) / 20
    
    volume_ratio_1d = np.full_like(volume_1d, np.nan)
    valid_vol = (~np.isnan(vol_ma_1d)) & (vol_ma_1d != 0)
    volume_ratio_1d[valid_vol] = volume_1d[valid_vol] / vol_ma_1d[valid_vol]
    
    # Align volume ratio to 12h timeframe
    volume_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 20)  # Need weekly EMA200, daily Donchian, daily volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or np.isnan(volume_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        volume_surge = volume_ratio_1d_aligned[i] > 2.0
        
        if position == 0:
            # Enter long: Weekly uptrend + price breaks above daily Donchian high + volume surge
            if weekly_uptrend and close[i] > donchian_high_1d_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly downtrend + price breaks below daily Donchian low + volume surge
            elif weekly_downtrend and close[i] < donchian_low_1d_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend turns down OR price breaks below daily Donchian low
            if not weekly_uptrend or close[i] < donchian_low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend turns up OR price breaks above daily Donchian high
            if not weekly_downtrend or close[i] > donchian_high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals