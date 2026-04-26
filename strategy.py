#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_ChopFilter_VolumeSpike
Hypothesis: Weekly Donchian channel breakouts capture major trend moves, while daily choppiness index and volume spike filters avoid false signals in ranging markets. Works in both bull and bear markets by trading breakouts in the direction of the weekly trend.
"""

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
    
    # Get weekly data for Donchian channel and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly trend: price vs 50-period EMA
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_uptrend = close > ema_50_1w_aligned
    weekly_downtrend = close < ema_50_1w_aligned
    
    # Daily choppiness index (14-period) for regime filter
    # Chop = 100 * log10(sum(TR) / (ATR * N)) / log10(N)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr / (atr * 14)) / np.log10(14)
    chop[np.isnan(chop)] = 50  # Default to middle when not enough data
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 14 for Chop, 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, weekly uptrend, low chop (trending), volume spike
            if (close[i] > donchian_high_aligned[i] and 
                weekly_uptrend[i] and 
                chop[i] < 40 and  # Trending market
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low, weekly downtrend, low chop (trending), volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_downtrend[i] and 
                  chop[i] < 40 and  # Trending market
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly Donchian low OR chop increases (ranging market)
            if close[i] < donchian_low_aligned[i] or chop[i] > 60:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly Donchian high OR chop increases (ranging market)
            if close[i] > donchian_high_aligned[i] or chop[i] > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_ChopFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0