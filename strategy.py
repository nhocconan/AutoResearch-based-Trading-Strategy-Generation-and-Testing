#!/usr/bin/env python3
# 6h_1d_RVI_Crossover_TrendFilter_Volume
# Hypothesis: 6s RVI (Relative Vigor Index) crossover with 1d trend filter and volume confirmation.
# RVI measures trend strength by comparing closing-open range to high-low range.
# Long when RVI crosses above its signal line in 1d uptrend with volume surge.
# Short when RVI crosses below signal line in 1d downtrend with volume surge.
# Designed for 6h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear by requiring 1d trend alignment, avoiding counter-trend whipsaws.
# Volume surge (2x 24-period MA) confirms institutional participation.

name = "6h_1d_RVI_Crossover_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RVI (Relative Vigor Index) for 6h
    # RVI = (Close - Open) / (High - Low) smoothed
    numerator = close - open_
    denominator = high - low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    raw_rvi = numerator / denominator
    
    # Smooth numerator and denominator separately (standard RVI method)
    # Num: (Close - Open) SMA
    # Den: (High - Low) SMA
    num_smooth = pd.Series(numerator).rolling(window=10, min_periods=10).mean().values
    den_smooth = pd.Series(denominator).rolling(window=10, min_periods=10).mean().values
    den_smooth = np.where(den_smooth == 0, 1e-10, den_smooth)
    rvi = num_smooth / den_smooth
    
    # Signal line: EMA of RVI
    rvi_signal = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Volume average (24-period for 6h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for RVI calculation + vol MA
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rvi[i]) or np.isnan(rvi_signal[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 1d close > EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # RVI crossover signals
        rvi_cross_up = rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1]
        rvi_cross_down = rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1]
        
        if position == 0:
            # Long: RVI crosses above signal line in uptrend with volume surge
            if rvi_cross_up and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: RVI crosses below signal line in downtrend with volume surge
            elif rvi_cross_down and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: RVI crosses below signal line or trend fails
                if rvi_cross_down or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: RVI crosses above signal line or trend fails
                if rvi_cross_up or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals