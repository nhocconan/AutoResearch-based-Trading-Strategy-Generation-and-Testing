#!/usr/bin/env python3
name = "6h_1d_MarketFacilitation_Index_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Market Facilitation Index (MFI) on daily data
    # MFI = (High - Low) / Volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Avoid division by zero
    mfi_1d = np.where(volume_1d != 0, (high_1d - low_1d) / volume_1d, 0.0)
    
    # Calculate 14-period EMA of MFI for trend direction
    mfi_series = pd.Series(mfi_1d)
    mfi_ema_14 = mfi_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily MFI EMA to 6h timeframe
    mfi_ema_14_aligned = align_htf_to_ltf(prices, df_1d, mfi_ema_14)
    
    # Calculate 6-period volume moving average for confirmation
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 6)  # Wait for MFI EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(mfi_ema_14_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: rising MFI (market facilitation increasing) with volume confirmation
            mfi_rising = mfi_ema_14_aligned[i] > mfi_ema_14_aligned[i-1]
            vol_condition = volume[i] > vol_ma_6[i] * 1.5
            
            if mfi_rising and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: falling MFI with volume confirmation
            elif not mfi_rising and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: MFI starts falling or volume drops significantly
            if not mfi_rising or volume[i] < vol_ma_6[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: MFI starts rising or volume drops significantly
            if mfi_rising or volume[i] < vol_ma_6[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Market Facilitation Index (MFI) trend with volume confirmation
# - MFI measures market efficiency: price movement per unit of volume
# - Rising MFI indicates efficient upward movement (strong buying)
# - Falling MFI indicates efficient downward movement (strong selling)
# - Volume confirmation ensures institutional participation
# - Works in both bull (rising MFI in uptrend) and bear (falling MFI in downtrend)
# - Exit when MFI direction changes or volume weakens
# - Position size 0.25 targets ~40-80 trades/year, avoiding fee drag
# - Uses actual daily MFI EMA for trend filter, avoiding look-ahead bias
# - Novel application of MFI as trend filter in 6m timeframe with 1d HTF