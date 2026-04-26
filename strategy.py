#!/usr/bin/env python3
"""
6h_ElderRay_Trend_VolumeConfirm_v1
Hypothesis: Elder Ray (Bull/Bear Power) combined with 1d EMA200 trend filter and volume confirmation (>2x average) captures strong directional moves with low false signals. Bull Power > 0 and Bear Power < 0 indicate underlying strength/weakness. Works in both bull and bear markets via 1d trend filter. Target 15-30 trades/year to minimize fee drag.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 13-period EMA for Elder Ray (standard)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # ATR(14) for volume spike threshold (dynamic)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 6h * 4 = 1 day)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA200(1d), EMA13, volume(24)
    start_idx = max(200, 13, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_200_val = ema_200_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_200_val) or np.isnan(avg_vol) or np.isnan(bull_val) or 
            np.isnan(bear_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long: Bull Power > 0 (buying strength) + 1d uptrend + volume
        long_condition = (bull_val > 0) and (close_val > ema_200_val) and volume_confirmed
        # Short: Bear Power < 0 (selling pressure) + 1d downtrend + volume
        short_condition = (bear_val < 0) and (close_val < ema_200_val) and volume_confirmed
        
        # Exit: Elder Ray divergence - Bull Power <= 0 for long, Bear Power >= 0 for short
        long_exit = (position == 1 and bull_val <= 0)
        short_exit = (position == -1 and bear_val >= 0)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0