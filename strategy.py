#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R(14) + volume spike + 1d ADX trend filter
# Long when Williams %R crosses above -20 (oversold bounce) AND volume > 1.5x 4h average AND 1d ADX > 25
# Short when Williams %R crosses below -80 (overbought rejection) AND volume > 1.5x 4h average AND 1d ADX > 25
# Williams %R captures momentum reversals, volume confirms conviction, ADX filters for trending markets
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Williams %R (14-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # === 4h Volume Spike (average volume) ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values  # 20 periods average
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # === 1d ADX trend filter (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d_arr - low_1d_arr
    tr2 = np.abs(high_1d_arr - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d_arr - np.roll(close_1d_arr, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d_arr - np.roll(high_1d_arr, 1)) > (np.roll(low_1d_arr, 1) - low_1d_arr), 
                       np.maximum(high_1d_arr - np.roll(high_1d_arr, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d_arr, 1) - low_1d_arr) > (high_1d_arr - np.roll(high_1d_arr, 1)), 
                        np.maximum(np.roll(low_1d_arr, 1) - low_1d_arr, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[np.isnan(dx)] = 0
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr_val = williams_r_aligned[i]
        vol_ma_val = vol_ma_4h_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 4h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # ADX filter: trending market (ADX > 25)
        trend_filter = adx_val > 25
        
        # Williams %R signals
        wr_oversold = wr_val < -80
        wr_overbought = wr_val > -20
        
        # === EXIT LOGIC (reverse signal) ===
        if position == 1:  # Long position
            # Exit long when Williams %R goes overbought (> -20)
            if wr_overbought:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit short when Williams %R goes oversold (< -80)
            if wr_oversold:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Williams %R crosses above -80 from oversold AND volume confirmation AND trend filter
            if wr_val > -80 and wr_oversold and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Williams %R crosses below -20 from overbought AND volume confirmation AND trend filter
            elif wr_val < -20 and wr_overbought and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR14_Volume1.5x_1dADX25_TrendFollow"
timeframe = "4h"
leverage = 1.0