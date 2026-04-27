#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: 1d strategy using Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Enter long when price closes above upper Donchian(20) with 1w uptrend (close > EMA50) and volume > 2.0x 20-period average.
Enter short when price closes below lower Donchian(20) with 1w downtrend (close < EMA50) and volume confirmation.
Exit on opposite Donchian level touch or 1w trend reversal (price crosses EMA50).
Designed for low trade frequency (~10-30/year) with discrete position sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by following the 1w trend while using Donchian channels for precise breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 1d timeframe (completed bars only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 1d
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian(20) (20) + 1w EMA50 (50) + volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1w EMA50 trend filter and volume spike
            # Long: price closes above upper Donchian AND above EMA50 (1w uptrend)
            long_condition = (close_val > upper_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below lower Donchian AND below EMA50 (1w downtrend)
            short_condition = (close_val < lower_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian (opposite level) OR 1w EMA50 turns bearish (price below EMA)
            if (close_val < lower_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches upper Donchian (opposite level) OR 1w EMA50 turns bullish (price above EMA)
            if (close_val > upper_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0