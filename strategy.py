#!/usr/bin/env python3
"""
4h_Trend_Following_With_1d_Volume_Spike_Rule
Hypothesis: Use 4h Donchian breakout (20) as primary signal, filtered by 1d volume spike (2x average volume) and 1d ADX trend filter (ADX > 25). Long when price breaks above Donchian high in bullish 1d trend with volume spike, short when breaks below Donchian low in bearish 1d trend with volume spike. Exit when price crosses opposite Donchian band. Designed for 4h to capture strong trending moves with low frequency (target 20-50 trades/year).
"""

name = "4h_Trend_Following_With_1d_Volume_Spike_Rule"
timeframe = "4h"
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
    
    # Get 1d data for filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(lookback, n):
        donch_high[i] = np.max(high[i-lookback:i])
        donch_low[i] = np.min(low[i-lookback:i])
    
    # 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_period = 14
    atr = smooth_wilder(tr, atr_period)
    dm_plus_smooth = smooth_wilder(dm_plus, atr_period)
    dm_minus_smooth = smooth_wilder(dm_minus, atr_period)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, atr_period)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume spike condition: current 1d volume > 2x 20-day average
    vol_spike = vol_1d > (2 * vol_ma_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend direction using ADX and price vs EMA proxy
        # Simplified: use ADX > 25 as trending, and price position for direction
        # We'll use close vs 20-period EMA on 1d as trend direction filter
        if i >= 20:  # Need enough for EMA
            # Simple trend: if ADX > 25 and price above recent average = uptrend
            # Use 20-period SMA on 1d as trend filter
            sma_20_1d = np.full_like(close_1d, np.nan)
            for j in range(20, len(close_1d)):
                sma_20_1d[j] = np.mean(close_1d[j-20:j])
            sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
            if not np.isnan(sma_20_1d_aligned[i]):
                trend_1d_up = adx_aligned[i] > 25 and close_1d_aligned[i] > sma_20_1d_aligned[i]
                trend_1d_down = adx_aligned[i] > 25 and close_1d_aligned[i] < sma_20_1d_aligned[i]
            else:
                trend_1d_up = False
                trend_1d_down = False
        else:
            trend_1d_up = False
            trend_1d_down = False
        
        if position == 0:
            # Long: Donchian breakout up in 1d uptrend with volume spike
            if (close[i] > donch_high[i] and 
                trend_1d_up and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down in 1d downtrend with volume spike
            elif (close[i] < donch_low[i] and 
                  trend_1d_down and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals