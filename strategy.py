#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Price breaks above/below daily Donchian channels (20-day high/low) with 1d EMA50 trend filter and volume confirmation. 
Donchian channels provide clear breakout levels in trending markets, while the 1d trend filter ensures alignment with daily momentum. 
Volume filters false breakouts. Works in bull/bear by trading only in direction of 1d trend. Target: 15-30 trades/year (60-120 total) to minimize fee drag.
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donch_high_20 = np.full(len(high_1d), np.nan)
    donch_low_20 = np.full(len(low_1d), np.nan)
    if len(high_1d) >= 20:
        for i in range(20-1, len(high_1d)):
            donch_high_20[i] = np.max(high_1d[i-20+1:i+1])
            donch_low_20[i] = np.min(low_1d[i-20+1:i+1])
    
    # Daily EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Daily volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align all indicators to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average daily volume (scaled)
        # 1d = 2 x 12h bars, so scale daily volume to 12h equivalent
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 2.0  # Average 12h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Donchian levels
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        price_above_donch_high = close[i] > donch_high_20_aligned[i]
        price_below_donch_low = close[i] < donch_low_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, in uptrend, with volume
            if price_above_donch_high and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, in downtrend, with volume
            elif price_below_donch_low and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below Donchian high or trend turns down
            if not price_above_donch_high or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above Donchian low or trend turns up
            if not price_below_donch_low or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals