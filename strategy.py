#!/usr/bin/env python3
"""
6h_Donchian_Breakout_1dTrend_Volume
Hypothesis: For 6s timeframe, use Donchian channel (20-period) breakouts filtered by 1-day EMA trend direction and volume spikes. Enters long when price breaks above upper Donchian band with bullish daily trend and volume confirmation; enters short when breaks below lower band with bearish daily trend and volume. Exits on opposite Donchian band touch. Designed to work in both bull and bear markets by following daily trend filter, avoiding counter-trend trades. Targets 15-25 trades/year via strict entry conditions requiring trend alignment, breakout, and volume confirmation.
"""

name = "6h_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily EMA Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Donchian Channel (20-period) on 6h ---
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above upper Donchian + bullish daily trend + volume
            if (close[i] > high_20[i] and 
                close[i] > ema_34_6h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + bearish daily trend + volume
            elif (close[i] < low_20[i] and 
                  close[i] < ema_34_6h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: touch opposite Donchian band
            if position == 1:
                # Exit long: price touches or goes below lower Donchian band
                if close[i] <= low_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes above upper Donchian band
                if close[i] >= high_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals