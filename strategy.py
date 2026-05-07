#!/usr/bin/env python3
"""
4H_Donchian20_12HTrend_VolumeConfirm
Hypothesis: 4H price breaks Donchian(20) high/low with 12H EMA50 trend confirmation and volume spike.
Works in bull/bear markets: Breakouts capture strong directional moves; EMA50 filter avoids counter-trend trades.
Volume confirmation ensures breakout strength. Targets 20-50 trades/year to minimize fee drag on 4H timeframe.
"""
name = "4H_Donchian20_12HTrend_VolumeConfirm"
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
    
    # Get 12H data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12H EMA50 for trend direction
    close_12h_series = pd.Series(df_12h['close'])
    ema_50 = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian(20) channels on 4H data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 4H volume > 1.8 x 30-period average volume
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 30)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 80 bars between trades (20 days on 4H TF) to reduce frequency
            if bars_since_exit < 80:
                continue
                
            # Long: price breaks above Donchian high with 12H EMA50 uptrend and volume spike
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.30
                position = 1
                bars_since_exit = 0
            # Short: price breaks below Donchian low with 12H EMA50 downtrend and volume spike
            elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and 
                  close[i] < ema_50_aligned[i] and volume_filter[i]):
                signals[i] = -0.30
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion)
            if position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals