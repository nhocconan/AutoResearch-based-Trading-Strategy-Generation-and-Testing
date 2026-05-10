#!/usr/bin/env python3
# 12h_Donchian_Breakout_1dTrend_Volume_Confirmed
# Hypothesis: 12h Donchian(20) breakout filtered by 1d EMA50 trend and volume surge.
# Uses price channel breakouts as primary signal with trend and volume confirmation.
# Targets 15-25 trades/year to minimize fee drag on 12h timeframe.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out).

name = "12h_Donchian_Breakout_1dTrend_Volume_Confirmed"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from 1d data (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average (24-period = 12 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Warmup: need Donchian (20) + EMA50 (50) + volume MA (24)
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_high = close[i] > donchian_high_aligned[i-1]
        breakdown_low = close[i] < donchian_low_aligned[i-1]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Donchian breakout with volume surge and 1d uptrend
            if breakout_high and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown with volume surge and 1d downtrend
            elif breakdown_low and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 4 bars (2 days)
            if bars_since_entry < 4:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below Donchian low or trend changes
                if close[i] < donchian_low_aligned[i-1] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above Donchian high or trend changes
                if close[i] > donchian_high_aligned[i-1] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals