#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1w Camarilla pivot breakout + volume confirmation + 1d EMA trend filter.
Long when price breaks above weekly R4 with volume > 1.5x 20-period 1d volume average and price > 1d EMA50.
Short when price breaks below weekly S4 with volume > 1.5x 20-period 1d volume average and price < 1d EMA50.
Weekly Camarilla pivots calculated from prior week's OHLC. Uses discrete position sizing 0.25.
Targets 12-37 trades/year (50-150 over 4 years) by requiring confluence of weekly structure, volume, and trend.
Works in bull markets (trend continuation via EMA filter) and bear markets (mean reversion at extreme weekly levels).
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
    
    # Get weekly data for Camarilla pivots (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (R3, R4, S3, S4) from prior week
    # Camarilla formula: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    weekly_range = high_1w - low_1w
    r4 = close_1w + 1.1 * weekly_range * 1.1 / 2
    r3 = close_1w + 1.1 * weekly_range * 1.1 / 4
    s3 = close_1w - 1.1 * weekly_range * 1.1 / 4
    s4 = close_1w - 1.1 * weekly_range * 1.1 / 2
    
    # Get daily data for EMA50 and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume and above 1d EMA50 (bullish bias)
            if (close[i] > r4_aligned[i] and 
                volume_confirmed and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume and below 1d EMA50 (bearish bias)
            elif (close[i] < s4_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly R3 (take profit at first support)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly S3 (take profit at first resistance)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wCamarilla_R3R4_S3S4_Volume_EMA50"
timeframe = "6h"
leverage = 1.0