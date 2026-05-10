#!/usr/bin/env python3
"""
1d_Weekly_Channel_Breakout_1wTrend
Hypothesis: Price breaks the 20-week high or low calculated from 1-week data, with 1-week EMA10 trend filter and volume confirmation. 
Breakouts from weekly extremes capture sustained momentum across market regimes, while the weekly trend filter ensures alignment with long-term direction. 
Volume confirms breakout strength. Designed for low trade frequency (<15/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_Weekly_Channel_Breakout_1wTrend"
timeframe = "1d"
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
    
    # 1-week data for multi-week extremes and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 20-week high/low from 1w data (approx 5 months)
    lookback = 20
    high_20w = np.full(len(high_1w), np.nan)
    low_20w = np.full(len(low_1w), np.nan)
    
    if len(high_1w) >= lookback:
        for i in range(lookback, len(high_1w)):
            high_20w[i] = np.max(high_1w[i-lookback:i])
            low_20w[i] = np.min(low_1w[i-lookback:i])
    
    # 1-week EMA10 for trend filter
    ema10_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[:10])
        alpha = 2 / (10 + 1)
        for i in range(10, len(close_1w)):
            ema10_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema10_1w[i-1]
    
    # 1-week volume SMA5 for volume confirmation
    vol_sma5_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 5:
        vol_sma5_1w[4] = np.mean(volume_1w[:5])
        for i in range(5, len(volume_1w)):
            vol_sma5_1w[i] = (vol_sma5_1w[i-1] * 4 + volume_1w[i]) / 5
    
    # Align 1w indicators to 1d
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    vol_sma5_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma5_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Wait for EMA10
    
    for i in range(start_idx, n):
        if np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_sma5_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1w volume (scaled)
        # 1w = 5 x 1d, so scale 1w volume to daily equivalent
        vol_1w_scaled = vol_sma5_1w_aligned[i] / 5.0  # Average daily-equivalent volume from 1w data
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        # Trend and price relative to 20-week levels
        is_uptrend = close[i] > ema10_1w_aligned[i]
        is_downtrend = close[i] < ema10_1w_aligned[i]
        price_above_20w_high = close[i] > high_20w_aligned[i]
        price_below_20w_low = close[i] < low_20w_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-week high, in uptrend, with volume
            if price_above_20w_high and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-week low, in downtrend, with volume
            elif price_below_20w_low and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below 20-week high or trend turns down
            if not price_above_20w_high or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above 20-week low or trend turns up
            if not price_below_20w_low or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals