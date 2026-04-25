#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h Supertrend Filter + Volume Confirmation
Hypothesis: Donchian breakouts capture strong momentum moves. 
12h Supertrend filter ensures we only trade in the direction of the higher timeframe trend, 
reducing whipsaws in sideways markets. Volume confirmation adds validity to breakouts.
Designed for 6h timeframe targeting 12-35 trades/year (50-140 over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull markets via 
breakout continuation and in bear markets via filtered mean-reversion from extreme levels.
"""

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
    
    # Get 12h data for Supertrend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation
    tr1 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = np.zeros(len(close_12h))
    atr_12h[:9] = np.nan
    for i in range(9, len(close_12h)):
        atr_12h[i] = np.mean(tr_12h[i-9:i+1])
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + 3.0 * atr_12h
    lower_band_12h = hl2_12h - 3.0 * atr_12h
    
    supertrend_12h = np.zeros(len(close_12h))
    direction_12h = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
    
    supertrend_12h[0] = upper_band_12h[0]
    direction_12h[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] <= supertrend_12h[i-1]:
            direction_12h[i] = -1
        else:
            direction_12h[i] = 1
        
        if direction_12h[i] == 1:
            supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
        else:
            supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
    
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate Donchian(20) channels for 6h
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, volume MA, and Supertrend to propagate
    start_idx = max(donchian_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(supertrend_12h_aligned[i]) or 
            np.isnan(direction_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        supertrend_val = supertrend_12h_aligned[i]
        direction_val = direction_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.8 * 20-period average
        volume_spike = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high AND 12h Supertrend uptrend AND volume spike
            long_condition = (curr_close > donchian_high_val) and (direction_val == 1) and volume_spike
            # Short: price breaks below Donchian low AND 12h Supertrend downtrend AND volume spike
            short_condition = (curr_close < donchian_low_val) and (direction_val == -1) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price breaks below Donchian low (reversal)
            # Calculate current ATR for stoploss
            if i >= 14:
                tr1 = np.abs(np.diff(close[max(0,i-14):i+1], prepend=close[max(0,i-14)]))
                tr2 = np.abs(high[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1))
                tr3 = np.abs(low[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1))
                tr2[0] = np.abs(high[max(0,i-14)] - close[max(0,i-14)])
                tr3[0] = np.abs(low[max(0,i-14)] - close[max(0,i-14)])
                tr_current = np.maximum(tr1, np.maximum(tr2, tr3))
                atr_current = np.mean(tr_current)
            else:
                atr_current = np.mean(high - low)  # fallback
            
            if curr_close <= entry_price - 2.5 * atr_current or curr_close < donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price breaks above Donchian high (reversal)
            if i >= 14:
                tr1 = np.abs(np.diff(close[max(0,i-14):i+1], prepend=close[max(0,i-14)]))
                tr2 = np.abs(high[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1))
                tr3 = np.abs(low[max(0,i-14):i+1] - np.roll(close[max(0,i-14):i+1], 1))
                tr2[0] = np.abs(high[max(0,i-14)] - close[max(0,i-14)])
                tr3[0] = np.abs(low[max(0,i-14)] - close[max(0,i-14)])
                tr_current = np.maximum(tr1, np.maximum(tr2, tr3))
                atr_current = np.mean(tr_current)
            else:
                atr_current = np.mean(high - low)  # fallback
            
            if curr_close >= entry_price + 2.5 * atr_current or curr_close > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hSupertrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0