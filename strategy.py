#!/usr/bin/env python3
"""
6h_VolumeBreakout_12hTrend_VolumeFilter
Hypothesis: Price breaks above/below the 60-period high/low (2.5-day range) on the 6h chart with volume > 2x 20-period average, only in the direction of the 12h EMA50 trend. Exits on trend reversal or when price returns to the 60-period midpoint. Designed to capture momentum bursts in both bull and bear markets by combining volatility breakouts with trend filtering and volume confirmation. Targets 15-30 trades/year via strict entry conditions requiring trend alignment, volatility expansion, and volume surge.
"""

name = "6h_VolumeBreakout_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 60-period High/Low (2.5-day range) ---
    high_60 = pd.Series(high).rolling(window=60, min_periods=60).max().values
    low_60 = pd.Series(low).rolling(window=60, min_periods=60).min().values
    mid_60 = (high_60 + low_60) / 2
    
    # --- 12h EMA50 Trend Filter ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
        if (np.isnan(high_60[i]) or np.isnan(low_60[i]) or 
            np.isnan(mid_60[i]) or np.isnan(ema_50_6h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above 60-period high + above 12h EMA50 + volume spike
            if (close[i] > high_60[i] and 
                close[i] > ema_50_6h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 60-period low + below 12h EMA50 + volume spike
            elif (close[i] < low_60[i] and 
                  close[i] < ema_50_6h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price returns to 60-period midpoint
                if (close[i] < ema_50_6h[i]) or \
                   (abs(close[i] - mid_60[i]) < (high_60[i] - low_60[i]) * 0.1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price returns to 60-period midpoint
                if (close[i] > ema_50_6h[i]) or \
                   (abs(close[i] - mid_60[i]) < (high_60[i] - low_60[i]) * 0.1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals