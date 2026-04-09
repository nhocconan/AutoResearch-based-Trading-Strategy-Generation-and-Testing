#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX filter and volume confirmation
# Uses 6h Donchian breakout with volume > 1.5x 24-period average
# Enters only when 1d ADX > 25 (trending market) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "6h_1d_adx_vol_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
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
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        dm_plus_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    atr_1d = np.zeros(len(df_1d))
    dm_plus_smooth_1d = np.zeros(len(df_1d))
    dm_minus_smooth_1d = np.zeros(len(df_1d))
    
    atr_1d[0] = tr_1d[0]
    dm_plus_smooth_1d[0] = dm_plus_1d[0]
    dm_minus_smooth_1d[0] = dm_minus_1d[0]
    
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
        dm_plus_smooth_1d[i] = (dm_plus_smooth_1d[i-1] * 13 + dm_plus_1d[i]) / 14
        dm_minus_smooth_1d[i] = (dm_minus_smooth_1d[i-1] * 13 + dm_minus_1d[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus_1d = np.zeros(len(df_1d))
    di_minus_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if atr_1d[i] > 0:
            di_plus_1d[i] = dm_plus_smooth_1d[i] / atr_1d[i] * 100
            di_minus_1d[i] = dm_minus_smooth_1d[i] / atr_1d[i] * 100
        else:
            di_plus_1d[i] = 0
            di_minus_1d[i] = 0
    
    # Calculate DX and ADX
    dx_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if di_plus_1d[i] + di_minus_1d[i] > 0:
            dx_1d[i] = abs(di_plus_1d[i] - di_minus_1d[i]) / (di_plus_1d[i] + di_minus_1d[i]) * 100
        else:
            dx_1d[i] = 0
    
    adx_1d = np.zeros(len(df_1d))
    adx_1d[0] = dx_1d[0]
    for i in range(1, len(df_1d)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    # Align ADX to 6h timeframe (only use completed daily bars)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Donchian channel (20-period)
    donch_high_6h = np.full(n, np.nan)
    donch_low_6h = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high_6h[i] = np.max(high[i-20:i])
        donch_low_6h[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 24-period average on 6h (4 days)
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
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_6h[i]) or 
            np.isnan(donch_low_6h[i]) or 
            np.isnan(adx_6h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending market (ADX > 25)
        if adx_6h[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h Donchian low
            if close[i] <= donch_low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h Donchian high
            if close[i] >= donch_high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 6h Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_24[i] if vol_ma_24[i] > 0 else 0
            if (close[i] > donch_high_6h[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 6h Donchian low with volume confirmation
            elif (close[i] < donch_low_6h[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals