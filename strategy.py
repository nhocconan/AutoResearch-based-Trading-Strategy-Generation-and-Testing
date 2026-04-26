#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20-period MA).
Long when price breaks above upper Donchian channel in 1d uptrend with volume spike.
Short when price breaks below lower Donchian channel in 1d downtrend with volume spike.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12-37 trades/year on 12h timeframe.
Works in both bull and bear markets by following the 1d trend.
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
    
    # Get 1d data for trend filter and Donchian calculation (prior completed daily candle)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from prior 20 completed 1d candles
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high of last 20 daily candles
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 daily candles
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (they change only when new 1d candle forms)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 50 for EMA, 20 for volume MA)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with 1d uptrend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with 1d downtrend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below lower Donchian (breakdown) OR 1d trend changes to downtrend
            if (close[i] < donchian_low_aligned[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above upper Donchian (breakout) OR 1d trend changes to uptrend
            if (close[i] > donchian_high_aligned[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0