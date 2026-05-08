#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChaikinMoneyFlow_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chaikin Money Flow
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chaikin Money Flow (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high_1d - low_1d
    mfm = np.zeros_like(hl_range)
    mask = hl_range != 0
    mfm[mask] = ((close_1d[mask] - low_1d[mask]) - (high_1d[mask] - close_1d[mask])) / hl_range[mask]
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume_1d
    
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    cmf_raw = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:  # 20-period minimum
            sum_mfv = np.sum(mfv[i-19:i+1])
            sum_volume = np.sum(volume_1d[i-19:i+1])
            if sum_volume != 0:
                cmf_raw[i] = sum_mfv / sum_volume
    
    # Align 1d CMF to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf_raw)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for CMF and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(cmf_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # CMF signal: positive for buying pressure, negative for selling pressure
        cmf_positive = cmf_aligned[i] > 0.05  # threshold to avoid noise
        cmf_negative = cmf_aligned[i] < -0.05
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                # Long when CMF shows strong buying pressure
                if cmf_positive:
                    signals[i] = 0.25
                    position = 1
                # Short when CMF shows strong selling pressure
                elif cmf_negative:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Manage long position
            exit_signal = False
            # Exit when CMF turns negative or volume confirmation lost
            if not cmf_positive:
                exit_signal = True
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            # Exit when CMF turns positive or volume confirmation lost
            if not cmf_negative:
                exit_signal = True
            elif not vol_confirmed:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals