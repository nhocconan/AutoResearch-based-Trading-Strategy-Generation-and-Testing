#!/usr/bin/env python3
"""
6h Weekly Pivot Point Breakout with Volume Confirmation
Strategy: Enter long when price breaks above weekly R3 with volume confirmation,
          short when price breaks below weekly S3 with volume confirmation.
          Use weekly trend (price vs weekly SMA200) to avoid counter-trend trades.
          Designed for low trade frequency with clear breakout edge in both bull and bear markets.
          Weekly pivots provide strong institutional levels that work across market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough history for weekly SMA200
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:  # Need enough weekly history for SMA200
        return np.zeros(n)
    
    # Calculate weekly high, low, close for pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Calculate weekly R3 and S3: R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r3 = weekly_high + 2.0 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2.0 * (weekly_high - weekly_pivot)
    
    # Calculate weekly SMA200 for trend filter
    sma_200_1w = pd.Series(weekly_close).rolling(window=200, min_periods=200).mean().values
    
    # Align weekly levels to 6h timeframe
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 200  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or
            np.isnan(weekly_close_aligned[i]) or
            np.isnan(sma_200_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_r3_level = weekly_r3_aligned[i]
        weekly_s3_level = weekly_s3_aligned[i]
        weekly_close_val = weekly_close_aligned[i]
        sma_200 = sma_200_1w_aligned[i]
        
        if position == 0:
            # Long: break above weekly R3 with volume spike and above weekly SMA200 (bullish trend)
            if (price > weekly_r3_level and volume_spike[i] and weekly_close_val > sma_200):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3 with volume spike and below weekly SMA200 (bearish trend)
            elif (price < weekly_s3_level and volume_spike[i] and weekly_close_val < sma_200):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below weekly S3 or weekly close drops below SMA200
            if price < weekly_s3_level or weekly_close_val < sma_200:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above weekly R3 or weekly close rises above SMA200
            if price > weekly_r3_level or weekly_close_val > sma_200:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0