#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1
Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter and volume confirmation (>1.5x average volume). Uses discrete position sizing (0.25) to minimize fee churn. Exits when price retests the broken Donchian level or reverses across the 1d EMA50. Works in both bull and bear markets by following the 1d trend direction, confirmed by volume to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA, Donchian, volume
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
    
    # Calculate Donchian channels from previous 4h bar (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian upper and lower bands (from previous bar to avoid look-ahead)
    donchian_upper = np.roll(high_roll, 1)
    donchian_lower = np.roll(low_roll, 1)
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 20 for Donchian and volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(upper) or 
            np.isnan(lower)):
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
        
        # Long logic: price breaks above Donchian upper with 1d uptrend and volume confirmation
        long_condition = (close_val > upper) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Donchian lower with 1d downtrend and volume confirmation
        short_condition = (close_val < lower) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: 
        # Long exit: price retests or breaks below Donchian upper (failed breakout) OR closes below 1d EMA (trend change)
        long_exit = (position == 1 and (close_val <= upper or close_val < ema_val))
        # Short exit: price retests or breaks above Donchian lower (failed breakout) OR closes above 1d EMA (trend change)
        short_exit = (position == -1 and (close_val >= lower or close_val > ema_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
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

name = "4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0