#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_1dTrend_VolumeFilter
Hypothesis: Elder Ray (Bull/Bear Power) identifies institutional accumulation/distribution. 
Enter long when Bull Power > 0 AND price breaks above prior 6h high with volume confirmation in 1d uptrend.
Enter short when Bear Power < 0 AND price breaks below prior 6h low with volume confirmation in 1d downtrend.
Uses 1d EMA34 for trend filter and 6h Donchian(20) for breakout levels. Volume > 1.5x 20-bar MA.
Designed for 12-37 trades/year (50-150 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following 1d trend.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h Donchian(20) for breakout levels
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for Donchian, 20 for vol, 13 for EMA13, 34 for 1d EMA)
    start_idx = max(20, 20, 13, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        vol_spike = volume_spike[i]
        bull_pwr = bull_power[i]
        bear_pwr = bear_power[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: 
        # Long: Bull Power > 0 (buying pressure) + break above Donchian high + volume + 1d uptrend
        # Short: Bear Power < 0 (selling pressure) + break below Donchian low + volume + 1d downtrend
        long_entry = (bull_pwr > 0) and (close_val > donchian_high) and vol_spike and bullish_1d
        short_entry = (bear_pwr < 0) and (close_val < donchian_low) and vol_spike and bearish_1d
        
        # Exit conditions: 
        # Long exit: Bull Power <= 0 OR price breaks below Donchian low OR 1d trend turns bearish
        # Short exit: Bear Power >= 0 OR price breaks above Donchian high OR 1d trend turns bullish
        exit_long = (bull_pwr <= 0) or (close_val < donchian_low) or not bullish_1d
        exit_short = (bear_pwr >= 0) or (close_val > donchian_high) or not bearish_1d
        
        # Minimum holding period: 2 bars
        min_hold = 2
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "6h_ElderRay_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0