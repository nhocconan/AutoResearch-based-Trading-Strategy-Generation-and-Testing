#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 12h breakout above/below 20-period Donchian channels in direction of 1w EMA50 trend, confirmed by volume spike (>1.5x 50-bar MA). Uses 1w HTF for trend alignment to capture major market direction, reducing whipsaws in both bull and bear markets. Volume confirmation filters low-momentum breakouts. Designed for 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25) to minimize fee drag. Works in ranging markets by requiring strong trend alignment and volume confirmation.
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for Donchian, 50 for EMA/volume)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Entry conditions: breakout of Donchian channel in trend direction with volume
        long_entry = (close_val > upper_channel) and bullish_1w and vol_spike
        short_entry = (close_val < lower_channel) and bearish_1w and vol_spike
        
        # Exit conditions: opposite Donchian channel touch (or trend reversal)
        exit_long = (close_val < lower_channel) or not bullish_1w
        exit_short = (close_val > upper_channel) or not bearish_1w
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0