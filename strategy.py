#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Uses 12h Camarilla R1/S1 levels for mean reversion entries in ranging markets, 
filtered by 12h EMA50 trend direction and volume spikes. In trending markets (price > EMA50),
buy at S1 support; in ranging markets, trade R1/S1 reversals. Designed for 4h timeframe 
to achieve 75-200 total trades over 4 years (19-50/year). Volume spike filter reduces 
false signals. Discrete position sizing (0.25) minimizes fee drag. Works in both bull 
and bear markets by adapting to 12h trend regime.
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
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1 = pivot + (range_12h * 1.0 / 12)
    s1 = pivot - (range_12h * 1.0 / 12)
    
    # Align 12h indicators to 4h timeframe (completed bars only)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 12h EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine market regime: trending if price > EMA50, ranging otherwise
            is_trending = close_val > ema_val
            
            if is_trending:
                # In uptrend: buy at S1 support with volume confirmation
                long_condition = (close_val <= s1_val) and vol_conf
                if long_condition:
                    signals[i] = size
                    position = 1
            else:
                # In ranging market: mean reversion at R1/S1 levels
                long_condition = (close_val <= s1_val) and vol_conf
                short_condition = (close_val >= r1_val) and vol_conf
                
                if long_condition:
                    signals[i] = size
                    position = 1
                elif short_condition:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long when price reaches midpoint (pivot) or trend breaks
            pivot_aligned = (r1_aligned[i] + s1_aligned[i]) / 2  # Approximate pivot
            exit_condition = (close_val >= pivot_aligned) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price reaches midpoint (pivot) or trend breaks
            pivot_aligned = (r1_aligned[i] + s1_aligned[i]) / 2  # Approximate pivot
            exit_condition = (close_val <= pivot_aligned) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0