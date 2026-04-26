#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeRegime
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume regime (ATR ratio > 1.2 = expansion). Only trade breakouts aligned with weekly trend during volatility expansion to capture strong moves while avoiding chop. Uses discrete sizing 0.25 to limit trades (~25/year). Volume regime ensures institutional participation.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(20) for Donchian and volume regime
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for ATR ratio and EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_regime = atr_ratio[i] > 1.2  # volatility expansion
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume regime AND aligned with 1w EMA50 trend
        long_entry = (close_val > upper_val) and vol_regime and (close_val > ema_50_val)
        short_entry = (close_val < lower_val) and vol_regime and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian middle line or trend reversal
            mid_channel = (upper_val + lower_val) / 2
            if close_val < mid_channel or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian middle line or trend reversal
            mid_channel = (upper_val + lower_val) / 2
            if close_val > mid_channel or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeRegime"
timeframe = "6h"
leverage = 1.0