#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume confirmation
# Uses 4h Donchian(20) breakout with volume > 1.5x 24-period average
# Only trades when 12h ADX > 25 (strong trend) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "4h_12h_adx_vol_breakout_v2"
timeframe = "4h"
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
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr0 = high_12h[i] - low_12h[i]
        tr1 = abs(high_12h[i] - close_12h[i-1])
        tr2 = abs(low_12h[i] - close_12h[i-1])
        tr_12h[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    dm_plus_12h = np.zeros(len(df_12h))
    dm_minus_12h = np.zeros(len(df_12h))
    for i in range(1, len(df_12h)):
        up_move = high_12h[i] - high_12h[i-1]
        down_move = low_12h[i-1] - low_12h[i]
        dm_plus_12h[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus_12h[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR and DM
    tr_14_12h = np.zeros(len(df_12h))
    dm_plus_14_12h = np.zeros(len(df_12h))
    dm_minus_14_12h = np.zeros(len(df_12h))
    
    for i in range(len(df_12h)):
        if i < 14:
            tr_14_12h[i] = np.nan
            dm_plus_14_12h[i] = np.nan
            dm_minus_14_12h[i] = np.nan
        elif i == 14:
            tr_14_12h[i] = np.nansum(tr_12h[1:15])
            dm_plus_14_12h[i] = np.nansum(dm_plus_12h[1:15])
            dm_minus_14_12h[i] = np.nansum(dm_minus_12h[1:15])
        else:
            tr_14_12h[i] = tr_14_12h[i-1] - (tr_14_12h[i-1] / 14) + tr_12h[i]
            dm_plus_14_12h[i] = dm_plus_14_12h[i-1] - (dm_plus_14_12h[i-1] / 14) + dm_plus_12h[i]
            dm_minus_14_12h[i] = dm_minus_14_12h[i-1] - (dm_minus_14_12h[i-1] / 14) + dm_minus_12h[i]
    
    # DI+ and DI-
    di_plus_12h = np.full(len(df_12h), np.nan)
    di_minus_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        if tr_14_12h[i] > 0:
            di_plus_12h[i] = (dm_plus_14_12h[i] / tr_14_12h[i]) * 100
            di_minus_12h[i] = (dm_minus_14_12h[i] / tr_14_12h[i]) * 100
    
    # DX and ADX
    dx_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        if di_plus_12h[i] + di_minus_12h[i] > 0:
            dx_12h[i] = (abs(di_plus_12h[i] - di_minus_12h[i]) / (di_plus_12h[i] + di_minus_12h[i])) * 100
    
    adx_12h = np.full(len(df_12h), np.nan)
    for i in range(27, len(df_12h)):  # 14 + 13 for smoothing
        if i == 27:
            adx_12h[i] = np.nanmean(dx_12h[14:28])
        elif not np.isnan(dx_12h[i]):
            adx_12h[i] = (adx_12h[i-1] * 13 + dx_12h[i]) / 14
    
    # Align 12h ADX to 4h timeframe (only use completed 12h bars)
    adx_12h_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 4h Donchian channel (20-period)
    donch_high_4h = np.full(n, np.nan)
    donch_low_4h = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high_4h[i] = np.max(high[i-20:i])
        donch_low_4h[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 24-period average on 4h (6 days)
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
        if (np.isnan(donch_high_4h[i]) or 
            np.isnan(donch_low_4h[i]) or 
            np.isnan(adx_12h_4h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in strong trend (ADX > 25)
        if adx_12h_4h[i] <= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low
            if close[i] <= donch_low_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high
            if close[i] >= donch_high_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_24[i] if vol_ma_24[i] > 0 else 0
            if (close[i] > donch_high_4h[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian low with volume confirmation
            elif (close[i] < donch_low_4h[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals