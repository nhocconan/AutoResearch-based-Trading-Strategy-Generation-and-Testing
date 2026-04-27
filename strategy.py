#!/usr/bin/env python3
"""
6h_Aggressive_Squeeze_Breakout_1dTrend_Volume
Hypothesis: Combines Bollinger Band squeeze detection with Donchian breakout on 6h, filtered by 1d trend and volume spike.
Works in bull markets via breakout continuation and in bear via mean-reversion off volatility contractions.
Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 1d
    close_1d = df_1d['close'].values
    bb_period = 20
    bb_std = 2.0
    sma_1d = np.full(len(close_1d), np.nan)
    std_1d = np.full(len(close_1d), np.nan)
    upper_bb = np.full(len(close_1d), np.nan)
    lower_bb = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= bb_period:
        for i in range(bb_period - 1, len(close_1d)):
            sma_1d[i] = np.mean(close_1d[i - bb_period + 1:i + 1])
            std_1d[i] = np.std(close_1d[i - bb_period + 1:i + 1])
            upper_bb[i] = sma_1d[i] + bb_std * std_1d[i]
            lower_bb[i] = sma_1d[i] - bb_std * std_1d[i]
    
    # Bollinger Band Width (normalized)
    bb_width = np.full(len(close_1d), np.nan)
    for i in range(bb_period - 1, len(close_1d)):
        if sma_1d[i] > 0:
            bb_width[i] = (upper_bb[i] - lower_bb[i]) / sma_1d[i]
    
    # Bollinger Band squeeze: BB width below 20-period percentile (20th percentile)
    bb_width_percentile = np.full(len(close_1d), np.nan)
    lookback = 20
    for i in range(lookback - 1, len(close_1d)):
        if not np.isnan(bb_width[i - lookback + 1:i + 1]).all():
            bb_width_percentile[i] = np.percentile(
                bb_width[i - lookback + 1:i + 1], 20
            )
    
    squeeze = np.full(len(close_1d), False)
    for i in range(bb_period - 1, len(close_1d)):
        if not np.isnan(bb_width[i]) and not np.isnan(bb_width_percentile[i]):
            squeeze[i] = bb_width[i] < bb_width_percentile[i]
    
    # 1d trend: EMA(50)
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i - 1] * (1 - multiplier))
    
    # Align 1d indicators to 6h
    squeeze_aligned = align_ltf_to_htf(prices, df_1d, squeeze)
    ema_aligned = align_ltf_to_htf(prices, df_1d, ema_1d)
    
    # 6h Donchian breakout (20-period)
    donch_period = 20
    upper_donch = np.full(n, np.nan)
    lower_donch = np.full(n, np.nan)
    
    if n >= donch_period:
        for i in range(donch_period - 1, n):
            upper_donch[i] = np.max(high[i - donch_period + 1:i + 1])
            lower_donch[i] = np.min(low[i - donch_period + 1:i + 1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Donchian (20), volume MA (20), 1d indicators
    start_idx = max(donch_period - 1, vol_ma_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(upper_donch[i]) or
            np.isnan(lower_donch[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Squeeze condition from 1d (must be in squeeze)
        in_squeeze = squeeze_aligned[i] if not np.isnan(squeeze_aligned[i]) else False
        
        if position == 0:
            # Long entry: price breaks above Donchian upper in uptrend, volume, and squeeze
            if price > upper_donch[i] and uptrend and volume_confirmation and in_squeeze:
                signals[i] = size
                position = 1
            # Short entry: price breaks below Donchian lower in downtrend, volume, and squeeze
            elif price < lower_donch[i] and downtrend and volume_confirmation and in_squeeze:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below Donchian lower or trend reverses
            if price < lower_donch[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns above Donchian upper or trend reverses
            if price > upper_donch[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Aggressive_Squeeze_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0