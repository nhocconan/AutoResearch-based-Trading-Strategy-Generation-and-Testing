#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for regime filter (trending when ADX > 25, ranging when ADX < 20).
- Entry: Long when Bull Power > 0 and Bear Power < 0 in 1d trending regime (ADX > 25) with volume > 1.5 * 6h volume MA(20); Short when Bear Power < 0 and Bull Power > 0 in 1d trending regime with volume confirmation.
- Exit: Opposite Elder Ray signal or volume drying up (volume < 0.5 * 6h volume MA(20)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Elder Ray measures bull/bear power via EMA13, ADX filter avoids whipsaws in ranging markets, volume confirmation ensures institutional participation. Works in both bull and bear markets by following the dominant trend as measured by Elder Ray.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA13 and ADX
        return np.zeros(n)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Bull Power and Bear Power
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate 1d ADX(14) for regime filter
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - np.roll(df_1d['high'], 1)) > (np.roll(df_1d['low'], 1) - df_1d['low']),
                       np.maximum(df_1d['high'] - np.roll(df_1d['high'], 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'], 1) - df_1d['low']) > (df_1d['high'] - np.roll(df_1d['high'], 1)),
                        np.maximum(np.roll(df_1d['low'], 1) - df_1d['low'], 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_volume = volume[i]
        
        # Volume confirmation: > 1.5x average for entry, < 0.5x for exit (volume drying up)
        vol_confirmed_entry = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        vol_confirmed_exit = curr_volume < 0.5 * vol_ma_6h_aligned[i]
        
        # Regime filter: ADX > 25 for trending market
        is_trending = adx_aligned[i] > 25
        
        # Elder Ray signals
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Long conditions: Bull Power > 0, Bear Power < 0, trending regime, volume confirmation
        long_entry = (bull_power_val > 0 and bear_power_val < 0 and 
                     is_trending and vol_confirmed_entry)
        
        # Short conditions: Bear Power < 0, Bull Power > 0, trending regime, volume confirmation
        short_entry = (bear_power_val < 0 and bull_power_val > 0 and 
                      is_trending and vol_confirmed_entry)
        
        # Exit conditions: opposite Elder Ray signal or volume drying up
        long_exit = (bull_power_val < 0 or bear_power_val > 0 or vol_confirmed_exit)
        short_exit = (bull_power_val > 0 or bear_power_val < 0 or vol_confirmed_exit)
        
        if position == 0:
            # Check for entry signals
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: check exit conditions
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0