#!/usr/bin/env python3

# 1d_Donchian20_Breakout_WeeklyTrend_Filter
# Hypothesis: Daily Donchian(20) breakout filtered by weekly ADX trend (>25) and volume spike (>2x 20-week average).
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 20-40 trades/year per symbol to avoid fee drag.

name = "1d_Donchian20_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
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
    
    # Get weekly data for filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Weekly ADX for trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder smoothing
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
    
    # Weekly volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full_like(vol_1w, np.nan)
    for i in range(20, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-20:i])
    
    # Align weekly indicators to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume spike condition: current weekly volume > 2x 20-week average
    vol_spike = vol_1w > (2 * vol_ma_1w)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 5  # Prevent overtrading (approx 5 days)
    
    start_idx = max(20, 30)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction using ADX and price vs 20-period SMA
        sma_20_1w = np.full_like(close_1w, np.nan)
        for j in range(20, len(close_1w)):
            sma_20_1w[j] = np.mean(close_1w[j-20:j])
        sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
        
        if not np.isnan(sma_20_1w_aligned[i]):
            trend_1w_up = adx_aligned[i] > 25 and close_1w_aligned[i] > sma_20_1w_aligned[i]
            trend_1w_down = adx_aligned[i] > 25 and close_1w_aligned[i] < sma_20_1w_aligned[i]
        else:
            trend_1w_up = False
            trend_1w_down = False
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Donchian breakout in weekly uptrend with volume spike
            if (close[i] > donchian_high[i] and 
                trend_1w_up and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Donchian breakdown in weekly downtrend with volume spike
            elif (close[i] < donchian_low[i] and 
                  trend_1w_down and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals