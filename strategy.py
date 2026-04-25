#!/usr/bin/env python3
"""
6h Elder Ray + 12h SuperTrend + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength. 
SuperTrend(12h, ATR=10, mult=3) provides reliable trend filter. Volume spike confirms participation.
Works in bull/bear by trend-following with dynamic power thresholds. Target: 12-37 trades/year.
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
    
    # Get 12h data for SuperTrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ATR for SuperTrend
    tr1_12h = df_12h['high'].values[1:] - df_12h['low'].values[1:]
    tr2_12h = np.abs(df_12h['high'].values[1:] - df_12h['close'].values[:-1])
    tr3_12h = np.abs(df_12h['low'].values[1:] - df_12h['close'].values[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 12h SuperTrend
    hl2_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    upper_12h = hl2_12h + (3 * atr_12h)
    lower_12h = hl2_12h - (3 * atr_12h)
    
    supertrend_12h = np.full_like(hl2_12h, np.nan)
    direction_12h = np.full_like(hl2_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2_12h)):
        if np.isnan(supertrend_12h[i-1]):
            supertrend_12h[i] = lower_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h := df_12h['close'].values[i]:
                if direction_12h[i-1] == 1:
                    supertrend_12h[i] = max(lower_12h[i], supertrend_12h[i-1])
                    if close_12h < supertrend_12h[i]:
                        direction_12h[i] = -1
                        supertrend_12h[i] = upper_12h[i]
                    else:
                        direction_12h[i] = 1
                else:
                    supertrend_12h[i] = min(upper_12h[i], supertrend_12h[i-1])
                    if close_12h > supertrend_12h[i]:
                        direction_12h[i] = 1
                        supertrend_12h[i] = lower_12h[i]
                    else:
                        direction_12h[i] = -1
    
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Calculate ATR for volatility (10-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 (13) + SuperTrend calculation
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(supertrend_12h_aligned[i]) or np.isnan(direction_12h_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema13_val = ema_13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        st_direction = direction_12h_aligned[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Dynamic Elder Ray thresholds based on ATR
        bull_threshold = 0.5 * atr_value
        bear_threshold = 0.5 * atr_value
        
        # Elder Ray signals
        bullish_energy = bull_val > bull_threshold
        bearish_energy = bear_val > bear_threshold
        
        # Exit conditions
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish energy or trend reversal
                if bearish_energy or st_direction == -1:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish energy or trend reversal
                if bullish_energy or st_direction == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Elder Ray energy + SuperTrend alignment + volume spike
        if position == 0:
            # Long: bullish energy AND SuperTrend uptrend
            long_condition = bullish_energy and (st_direction == 1) and volume_spike
            # Short: bearish energy AND SuperTrend downtrend
            short_condition = bearish_energy and (st_direction == -1) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_SuperTrend12h_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0