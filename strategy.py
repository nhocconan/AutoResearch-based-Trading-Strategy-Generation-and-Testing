#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot trend and volume spikes capture high-probability moves.
Uses weekly Camarilla pivot levels (R1/S1) from prior week for trend filter (price > weekly R1 = uptrend, < weekly S1 = downtrend).
Volume confirmation: current volume > 2.0 * 20-period average.
Discrete sizing (0.25) to control fee drag. Target: 50-150 total trades over 4 years.
Works in bull (breakouts with trend) and bear (fade at extremes in range) via weekly pivot structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels (trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels (R1, S1) from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    range_1w = high_1w - low_1w
    camarilla_r1 = close_1w + 1.125 * range_1w
    camarilla_s1 = close_1w - 1.125 * range_1w
    
    # Get daily data for volume confirmation (optional, but use for stability)
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian(20) channels on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to primary timeframe (6h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)  # align weekly Donchian for stability
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)  # volume is LTF, but confirm using 1d avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need weekly Camarilla (1), Donchian (20), volume avg (20)
    start_idx = max(1, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs weekly Camarilla R1/S1
            uptrend = close_val > r1
            downtrend = close_val < s1
            
            if uptrend and vol_conf:
                # Long bias: long when price breaks above Donchian high with volume and uptrend
                if close_val > dch_high:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short bias: short when price breaks below Donchian low with volume and downtrend
                if close_val < dch_low:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (3*ATR) or Donchian low touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 3.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < dch_low:  # Donchian low touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (3*ATR) or Donchian high touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 3.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > dch_high:  # Donchian high touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0