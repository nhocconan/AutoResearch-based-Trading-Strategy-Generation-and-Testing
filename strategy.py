#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Uses 12h Donchian breakout with volume > 1.5x 24-period average
# Enters only when 1d ADX > 25 (trending market) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 15-35 trades/year per symbol to minimize fee drag

name = "12h_1d_adx_vol_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_high_12h = np.full(len(df_12h), np.nan)
    donch_low_12h = np.full(len(df_12h), np.nan)
    
    for i in range(20, len(df_12h)):
        donch_high_12h[i] = np.max(high_12h[i-20:i])
        donch_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Align 12h Donchian to 12h timeframe (identity alignment)
    donch_high_12h_aligned = donch_high_12h
    donch_low_12h_aligned = donch_low_12h
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    dm_plus_1d = np.zeros(len(df_1d))
    dm_minus_1d = np.zeros(len(df_1d))
    dm_plus_1d[0] = 0
    dm_minus_1d[0] = 0
    
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus_1d[i] = up_move
        else:
            dm_plus_1d[i] = 0
            
        if down_move > up_move and down_move > 0:
            dm_minus_1d[i] = down_move
        else:
            dm_minus_1d[i] = 0
    
    # Smoothed TR, DM+
    tr_14 = np.zeros(len(df_1d))
    dm_plus_14 = np.zeros(len(df_1d))
    dm_minus_14 = np.zeros(len(df_1d))
    
    tr_14[0] = tr_1d[0]
    dm_plus_14[0] = dm_plus_1d[0]
    dm_minus_14[0] = dm_minus_1d[0]
    
    for i in range(1, len(df_1d)):
        tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_1d[i]
        dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus_1d[i]
        dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus_1d[i]
    
    # Directional Indicators
    di_plus_1d = np.zeros(len(df_1d))
    di_minus_1d = np.zeros(len(df_1d))
    
    for i in range(14, len(df_1d)):
        if tr_14[i] > 0:
            di_plus_1d[i] = (dm_plus_14[i] / tr_14[i]) * 100
            di_minus_1d[i] = (dm_minus_14[i] / tr_14[i]) * 100
        else:
            di_plus_1d[i] = 0
            di_minus_1d[i] = 0
    
    # DX and ADX
    dx_1d = np.zeros(len(df_1d))
    adx_1d = np.zeros(len(df_1d))
    
    for i in range(14, len(df_1d)):
        di_sum = di_plus_1d[i] + di_minus_1d[i]
        if di_sum > 0:
            dx_1d[i] = abs(di_plus_1d[i] - di_minus_1d[i]) / di_sum * 100
        else:
            dx_1d[i] = 0
    
    # Smoothed DX for ADX
    adx_1d[0] = 0
    for i in range(1, len(df_1d)):
        if i < 14:
            adx_1d[i] = 0
        else:
            if i == 14:
                adx_1d[i] = np.mean(dx_1d[14:i+1])
            else:
                adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    # Align 1d ADX to 12h timeframe (only use completed daily bars)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 24-period average on 12h (12 days)
    vol_ma_24 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_12h_aligned[i]) or 
            np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(adx_12h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending market (ADX > 25)
        if adx_12h[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low
            if close[i] <= donch_low_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high
            if close[i] >= donch_high_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 12h Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_24[i] if vol_ma_24[i] > 0 else 0
            if (close[i] > donch_high_12h_aligned[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h Donchian low with volume confirmation
            elif (close[i] < donch_low_12h_aligned[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals