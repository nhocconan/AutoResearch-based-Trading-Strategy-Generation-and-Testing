#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX filter
# Uses 12h Donchian breakout for trend entry, confirmed by 1d volume spike (>2x 20-period avg)
# Enters only when 1d ADX > 25 (trending market) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "12h_1d_donchian_volume_adx_v1"
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
    
    # Align 12h Donchian to 12h timeframe (only use completed 12h bars)
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
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
    plus_dm_1d = np.zeros(len(df_1d))
    minus_dm_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR, +DM, -DM
    tr_sum_1d = np.zeros(len(df_1d))
    plus_dm_sum_1d = np.zeros(len(df_1d))
    minus_dm_sum_1d = np.zeros(len(df_1d))
    
    # Initial values (first 14 periods)
    for i in range(14):
        if i == 0:
            tr_sum_1d[i] = tr_1d[i]
            plus_dm_sum_1d[i] = plus_dm_1d[i]
            minus_dm_sum_1d[i] = minus_dm_1d[i]
        else:
            tr_sum_1d[i] = tr_sum_1d[i-1] + tr_1d[i]
            plus_dm_sum_1d[i] = plus_dm_sum_1d[i-1] + plus_dm_1d[i]
            minus_dm_sum_1d[i] = minus_dm_sum_1d[i-1] + minus_dm_1d[i]
    
    # Wilder's smoothing (after 14 periods)
    tr_14_1d = np.zeros(len(df_1d))
    plus_dm_14_1d = np.zeros(len(df_1d))
    minus_dm_14_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i < 14:
            tr_14_1d[i] = tr_sum_1d[i]
            plus_dm_14_1d[i] = plus_dm_sum_1d[i]
            minus_dm_14_1d[i] = minus_dm_sum_1d[i]
        elif i == 14:
            tr_14_1d[i] = tr_sum_1d[i]
            plus_dm_14_1d[i] = plus_dm_sum_1d[i]
            minus_dm_14_1d[i] = minus_dm_sum_1d[i]
        else:
            tr_14_1d[i] = tr_14_1d[i-1] - (tr_14_1d[i-1] / 14) + tr_1d[i]
            plus_dm_14_1d[i] = plus_dm_14_1d[i-1] - (plus_dm_14_1d[i-1] / 14) + plus_dm_1d[i]
            minus_dm_14_1d[i] = minus_dm_14_1d[i-1] - (minus_dm_14_1d[i-1] / 14) + minus_dm_1d[i]
    
    # Directional Indicators
    plus_di_1d = np.zeros(len(df_1d))
    minus_di_1d = np.zeros(len(df_1d))
    dx_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if tr_14_1d[i] > 0:
            plus_di_1d[i] = (plus_dm_14_1d[i] / tr_14_1d[i]) * 100
            minus_di_1d[i] = (minus_dm_14_1d[i] / tr_14_1d[i]) * 100
            dx_1d[i] = (abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])) * 100
        else:
            plus_di_1d[i] = 0
            minus_di_1d[i] = 0
            dx_1d[i] = 0
    
    # ADX (smoothed DX)
    adx_1d = np.zeros(len(df_1d))
    adx_sum = 0
    
    for i in range(len(df_1d)):
        if i < 27:  # First 14 + 13 for smoothing
            if i >= 14:
                adx_sum += dx_1d[i]
            if i >= 27:
                adx_1d[i] = adx_sum / 14
        else:
            adx_sum = adx_sum - (adx_sum / 14) + dx_1d[i]
            adx_1d[i] = adx_sum
    
    # Align 1d ADX to 12h timeframe (only use completed daily bars)
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 20-period average on 1d
    vol_ma_20_1d = np.zeros(len(df_1d))
    vol_sum = 0.0
    for i in range(len(df_1d)):
        vol_sum += df_1d['volume'].iloc[i]
        if i >= 20:
            vol_sum -= df_1d['volume'].iloc[i-20]
        if i >= 19:
            vol_ma_20_1d[i] = vol_sum / 20
    
    # Align volume MA to 12h timeframe
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_12h_aligned[i]) or 
            np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending market (ADX > 25)
        if adx_12h_aligned[i] <= 25:
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
            # Get current volume
            vol_current = volume[i]
            vol_ma = vol_ma_20_12h_aligned[i]
            
            # Avoid division by zero
            if vol_ma <= 0:
                vol_ratio = 0
            else:
                vol_ratio = vol_current / vol_ma
            
            # Enter long: price closes above 12h Donchian high with volume confirmation
            if (close[i] > donch_high_12h_aligned[i] and 
                vol_ratio > 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h Donchian low with volume confirmation
            elif (close[i] < donch_low_12h_aligned[i] and 
                  vol_ratio > 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals