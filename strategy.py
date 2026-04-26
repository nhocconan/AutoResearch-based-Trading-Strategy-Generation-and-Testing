#!/usr/bin/env python3
"""
6h_ElderRay_Trend_VolumeConfirm
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d EMA50 trend filter and volume confirmation (>1.5x average volume). 
Long when Bull Power > 0 and rising (2-bar momentum) with 1d uptrend and volume confirmation.
Short when Bear Power > 0 and rising (2-bar momentum) with 1d downtrend and volume confirmation.
Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear markets by following the 1d trend direction, confirmed by volume to avoid false signals.
Target: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300 total.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA, volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # 2-bar momentum for Elder Ray (rising power)
    bull_power_mom = bull_power - np.roll(bull_power, 2)
    bear_power_mom = bear_power - np.roll(bear_power, 2)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50 for 1d EMA, 13 for EMA13, 20 for volume, 2 for momentum)
    start_idx = max(50, 13, 20) + 2
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        ema_val = ema_50_1d_aligned[i]
        ema13_val = ema_13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        bull_power_mom_val = bull_power_mom[i]
        bear_power_mom_val = bear_power_mom[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(ema13_val) or np.isnan(bull_power_val) or 
            np.isnan(bear_power_val) or np.isnan(bull_power_mom_val) or np.isnan(bear_power_mom_val) or
            np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: Bull Power > 0 and rising (momentum positive) with 1d uptrend and volume confirmation
        long_condition = (bull_power_val > 0) and (bull_power_mom_val > 0) and (close_val > ema_val) and volume_confirmed
        # Short logic: Bear Power > 0 and rising (momentum positive) with 1d downtrend and volume confirmation
        short_condition = (bear_power_val > 0) and (bear_power_mom_val > 0) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: 
        # Long exit: Bull Power becomes negative OR closes below 1d EMA (trend change)
        long_exit = (position == 1 and (bull_power_val <= 0 or close_val < ema_val))
        # Short exit: Bear Power becomes negative OR closes above 1d EMA (trend change)
        short_exit = (position == -1 and (bear_power_val <= 0 or close_val > ema_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ElderRay_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0