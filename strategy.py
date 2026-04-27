#!/usr/bin/env python3
"""
1d_Pullback_to_EMA34_with_VolumeSpike
Hypothesis: Price often pulls back to the weekly EMA34 during trends. 
Entry: price touches EMA34 with volume spike, in direction of weekly trend (EMA50 slope).
Exit: price closes beyond EMA34 opposite direction or trailing stop.
Designed for fewer trades (~15-25/year) to avoid fee drag on 1d timeframe.
Works in bull via bounces off rising EMA, in bear via bounces off falling EMA.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter (slope)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_slope = ema_50_1w - np.roll(ema_50_1w, 1)
    ema_50_1w_slope[0] = 0
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_slope)
    
    # Calculate daily EMA34 for pullback zone
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: volume > 2.0 * 20-day average (fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_50_1w_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema34_val = ema_34_1d_aligned[i]
        ema50_val = ema_50_1w_aligned[i]
        ema50_slope = ema_50_1w_slope_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price touches EMA34 from below with volume spike and weekly uptrend
            if low[i] <= ema34_val <= high[i] and vol_spike_val and ema50_slope > 0:
                # Additional confirmation: price close above EMA34
                if close[i] > ema34_val:
                    signals[i] = size
                    position = 1
            # Short: price touches EMA34 from above with volume spike and weekly downtrend
            elif low[i] <= ema34_val <= high[i] and vol_spike_val and ema50_slope < 0:
                # Additional confirmation: price close below EMA34
                if close[i] < ema34_val:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price closes below EMA34 (failed hold) or weekly trend turns down
            if close[i] < ema34_val or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above EMA34 (failed hold) or weekly trend turns up
            if close[i] > ema34_val or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Pullback_to_EMA34_with_VolumeSpike"
timeframe = "1d"
leverage = 1.0