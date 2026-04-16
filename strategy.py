#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme readings with volume confirmation and ADX trend filter.
# Long when Williams %R < -80 (oversold) + volume > 1.5x 20-period average + ADX > 25 (trending).
# Short when Williams %R > -20 (overbought) + volume > 1.5x 20-period average + ADX > 25.
# Exit when Williams %R returns to neutral range (-50) or volume drops below average.
# Uses discrete position size 0.25. Williams %R identifies exhaustion points in both bull and bear markets,
# volume confirms conviction, ADX ensures we trade only in trending conditions to avoid chop.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13 or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # neutral
        else:
            williams_r[i] = ((highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i])) * -100
    
    # === 1d Indicators: ADX (14-period) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 4h data once before loop for volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_val = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to neutral (> -50) or ADX weakens (< 20)
            if williams_val > -50 or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to neutral (< -50) or ADX weakens (< 20)
            if williams_val < -50 or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: ADX > 25 (strong trend)
            trend_filter = adx_val > 25
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # LONG: Williams %R oversold (< -80), volume spike, strong trend
            if (williams_val < -80) and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R overbought (> -20), volume spike, strong trend
            elif (williams_val > -20) and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dWilliamsR_ADX_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0