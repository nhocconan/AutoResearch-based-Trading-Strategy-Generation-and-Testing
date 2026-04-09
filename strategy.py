#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Uses 4h Donchian(20) breakout with volume > 1.5x 24-period average
# Enters only when 1d ADX > 25 (trending market) to avoid chop
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 20-40 trades/year per symbol to minimize fee drag

name = "4h_1d_adx_vol_filter_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    up_move = np.zeros(len(df_1d))
    down_move = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move[i] = high_1d[i] - high_1d[i-1]
        down_move[i] = low_1d[i-1] - low_1d[i]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = np.zeros(len(df_1d))
    plus_dm_14 = np.zeros(len(df_1d))
    minus_dm_14 = np.zeros(len(df_1d))
    
    tr_14[0] = np.sum(tr_1d[:14]) if len(tr_1d) >= 14 else 0
    plus_dm_14[0] = np.sum(plus_dm[:14]) if len(plus_dm) >= 14 else 0
    minus_dm_14[0] = np.sum(minus_dm[:14]) if len(minus_dm) >= 14 else 0
    
    for i in range(1, len(df_1d)):
        tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_1d[i]
        plus_dm_14[i] = plus_dm_14[i-1] - (plus_dm_14[i-1] / 14) + plus_dm[i]
        minus_dm_14[i] = minus_dm_14[i-1] - (minus_dm_14[i-1] / 14) + minus_dm[i]
    
    # Avoid division by zero
    plus_di_14 = np.where(tr_14 != 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di_14 = np.where(tr_14 != 0, 100 * minus_dm_14 / tr_14, 0)
    
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14), 0)
    
    adx_1d = np.zeros(len(df_1d))
    adx_1d[0] = np.mean(dx[:14]) if len(dx) >= 14 else 0
    for i in range(1, len(df_1d)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h timeframe (only use completed daily bars)
    adx_14_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
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
    
    for i in range(28, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donch_high_4h[i]) or 
            np.isnan(donch_low_4h[i]) or 
            np.isnan(adx_14_4h[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending market (ADX > 25)
        if adx_14_4h[i] <= 25:
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