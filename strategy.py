#!/usr/bin/env python3
# 12h_1w1d_PriceChannel_Squeeze_Breakout
# Hypothesis: Combines weekly Donchian channel breakouts with daily volatility squeeze and volume confirmation
# to capture explosive moves in both bull and bear markets. Uses weekly trend filter to avoid counter-trend trades.
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.

name = "12h_1w1d_PriceChannel_Squeeze_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-period) for breakout signals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels
    upper_20 = np.full_like(high_1w, np.nan)
    lower_20 = np.full_like(low_1w, np.nan)
    
    for i in range(19, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-19:i+1])
        lower_20[i] = np.min(low_1w[i-19:i+1])
    
    # Align weekly Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Daily volatility squeeze (Bollinger Band Width < 20th percentile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2.0)
    ma_20 = np.full_like(close_1d, np.nan)
    std_20 = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        ma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    upper_bb = ma_20 + (2.0 * std_20)
    lower_bb = ma_20 - (2.0 * std_20)
    bb_width = (upper_bb - lower_bb) / ma_20
    
    # Calculate 50-period percentile of BB width (20th percentile threshold)
    bb_width_pct = np.full_like(bb_width, np.nan)
    for i in range(49, len(bb_width)):
        if not np.isnan(bb_width[i]):
            bb_width_pct[i] = np.percentile(bb_width[max(0, i-49):i+1], 20)
    
    squeeze = bb_width < bb_width_pct
    
    # Align daily squeeze to 12h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    
    # Daily volume confirmation (1.5x 20-period average)
    vol_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_20[i] = np.mean(volume[i-19:i+1])
    
    volume_confirm = volume > (1.5 * vol_20)
    
    # Weekly trend filter (price above/below 50-period EMA)
    ema50_1w = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):
        if i == 49:
            ema50_1w[i] = np.mean(close_1w[:50])
        else:
            ema50_1w[i] = (close_1w[i] * 2/51) + (ema50_1w[i-1] * 49/51)
    
    # Align weekly EMA to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    weekly_uptrend = close_1w > ema50_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend = close_1w < ema50_1w
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper Donchian during volatility squeeze with volume confirmation and weekly uptrend
            if (high[i] > upper_20_aligned[i] and
                squeeze_aligned[i] > 0.5 and
                volume_confirm[i] and
                weekly_uptrend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower Donchian during volatility squeeze with volume confirmation and weekly downtrend
            elif (low[i] < lower_20_aligned[i] and
                  squeeze_aligned[i] > 0.5 and
                  volume_confirm[i] and
                  weekly_downtrend_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below weekly lower Donchian or weekly trend turns down
            if (low[i] < lower_20_aligned[i] or
                weekly_uptrend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above weekly upper Donchian or weekly trend turns up
            if (high[i] > upper_20_aligned[i] or
                weekly_downtrend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals