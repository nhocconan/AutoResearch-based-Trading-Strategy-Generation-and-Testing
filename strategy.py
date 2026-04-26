#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts only when aligned with weekly pivot bias (above/below weekly pivot) and 1d EMA50 trend, with volume confirmation (>1.5x median). Weekly pivot provides institutional reference, Donchian captures breakouts, and 1d EMA50 ensures we trade with the daily trend. Volume confirmation filters low-conviction breakouts. Designed for BTC/ETH to work in bull/bear by requiring 1d trend alignment. Target: 12-30 trades/year (50-120 over 4 years) to avoid fee drag.
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
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get weekly data for pivot calculation (weekly HLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Get 6h data for Donchian calculation (need 20 periods)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Donchian(20) from previous 6h bar (to avoid look-ahead)
    donch_high = pd.Series(df_6h['high'].values).shift(1).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_6h['low'].values).shift(1).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x median volume (30-period)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d (50), weekly pivot (2), Donchian (20+1), volume median (30)
    start_idx = max(50, 2, 21, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Weekly pivot bias: above pivot = bullish bias, below pivot = bearish bias
        bullish_bias = close_val > weekly_pivot_val
        bearish_bias = close_val < weekly_pivot_val
        
        # Volume confirmation: only trade in above-average volume environments
        volume_confirm = volume_val > 1.5 * vol_median_val
        
        if position == 0:
            # Long: break above Donchian high with bullish bias from weekly pivot, uptrend, and volume confirmation
            long_signal = (high_val > donch_high_val) and \
                          bullish_bias and \
                          uptrend and \
                          volume_confirm
            
            # Short: break below Donchian low with bearish bias from weekly pivot, downtrend, and volume confirmation
            short_signal = (low_val < donch_low_val) and \
                           bearish_bias and \
                           downtrend and \
                           volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long until break below Donchian low (reversal signal)
            signals[i] = 0.25
            if low_val < donch_low_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short until break above Donchian high (reversal signal)
            signals[i] = -0.25
            if high_val > donch_high_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0