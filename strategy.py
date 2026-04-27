#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_1wTrend_Volume
Hypothesis: Weekly Donchian breakout above/below weekly channel with weekly trend alignment and volume confirmation. 
Captures major trend continuations while avoiding false breakouts in sideways markets. 
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian upper/lower (20-period)
    donchian_high_20 = np.full(len(high_1w), np.nan)
    donchian_low_20 = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high_20[i] = np.max(high_1w[i-20:i])
        donchian_low_20[i] = np.min(low_1w[i-20:i])
    
    # Weekly EMA(50) for trend filter
    ema_50 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50[i] = close_1w[i] * 0.04 + ema_50[i-1] * 0.96  # 2/(50+1) = 0.04
    
    # Weekly volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = np.full(len(vol_1w), np.nan)
    for i in range(20, len(vol_1w)):
        vol_ma_20_1w[i] = np.mean(vol_1w[i-20:i])
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Daily volume average (20-period) for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need weekly Donchian (20) + daily volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        vol_ratio_1w = vol_1w[min(i // 7, len(vol_1w)-1)] / vol_ma_20_1w_aligned[i] if vol_ma_20_1w_aligned[i] > 0 else 0
        
        # Volume confirmation: daily volume > 1.5x average OR weekly volume > 1.5x average
        volume_confirmation = vol_ratio > 1.5 or vol_ratio_1w > 1.5
        
        # Trend filter: price above/below weekly EMA(50)
        uptrend = price > ema_50_aligned[i]
        downtrend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above weekly Donchian high with volume and uptrend
            if volume_confirmation and uptrend and price > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume and downtrend
            elif volume_confirmation and downtrend and price < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to weekly Donchian midpoint or trend breaks
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price < donchian_mid or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to weekly Donchian midpoint or trend breaks
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price > donchian_mid or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_WeeklyDonchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0