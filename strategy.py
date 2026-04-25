#!/usr/bin/env python3
"""
6h Elder Force Index + 1d EMA34 Trend + Volume Spike
Hypothesis: Elder Force Index (EFI) combines price change and volume to measure buying/selling pressure.
Breakouts occur when EFI crosses zero with volume confirmation and higher timeframe trend alignment.
Works in bull markets (buy on positive EFI crosses) and bear markets (sell on negative EFI crosses).
Designed for 6h timeframe with 50-150 total trades over 4 years to balance opportunity and fee drag.
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
    
    # Get daily data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 days for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Force Index (EFI) on 6h: (close - close_prev) * volume
    # EFI > 0 = buying pressure, EFI < 0 = selling pressure
    close_series = pd.Series(close)
    efi = (close_series - close_series.shift(1)) * volume
    efi = efi.values  # Convert to numpy array
    
    # Calculate 20-period EFI MA for smoothing (to reduce noise)
    efi_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        efi_ma_20[i] = np.mean(efi[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EFI MA, volume MA, and EMA34
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(efi_ma_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_efi_ma = efi_ma_20[i]
        curr_efi = efi[i]  # Current raw EFI for zero-cross detection
        prev_efi = efi[i-1]  # Previous EFI for zero-cross detection
        vol_ma = vol_ma_20[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = volume[i] > 1.8 * vol_ma
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Zero-cross detection for EFI
        efi_cross_above = (prev_efi <= 0) and (curr_efi > 0)
        efi_cross_below = (prev_efi >= 0) and (curr_efi < 0)
        
        if position == 0:
            # Look for entry signals
            # Long: EFI crosses above zero with volume confirmation in uptrend
            long_entry = efi_cross_above and volume_confirm and uptrend
            # Short: EFI crosses below zero with volume confirmation in downtrend
            short_entry = efi_cross_below and volume_confirm and downtrend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: EFI crosses below zero OR price closes below 1d EMA34
            if efi_cross_below or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EFI crosses above zero OR price closes above 1d EMA34
            if efi_cross_above or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Elder_Force_Index_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0