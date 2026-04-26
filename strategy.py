#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar MA). Donchian channels provide clear breakout levels while 12h EMA50 ensures we trade with the higher timeframe trend. Volume spike filter reduces false breakouts. Designed for 4h timeframe to achieve 20-50 trades/year. Works in bull markets (breakouts with trend) and bear markets (short breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels on primary timeframe
    # Upper channel: 20-period high
    # Lower channel: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for Donchian/vol, 50 for 12h EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # Entry conditions: breakout of Donchian channel in trend direction with volume spike
        long_entry = (close_val > upper) and bullish_12h and vol_spike
        short_entry = (close_val < lower) and bearish_12h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on break of opposite channel or trend change
            if close_val < lower or not bullish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on break of opposite channel or trend change
            if close_val > upper or not bearish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0