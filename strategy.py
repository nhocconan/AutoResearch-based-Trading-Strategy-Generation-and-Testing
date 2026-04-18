#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high on 4h, ADX(14) on 1d > 25 (trending),
# and volume > 1.5x 20-period average. Short when price breaks below Donchian(20) low with same conditions.
# Exit when price returns to Donchian midpoint. Designed for ~20-40 trades/year per symbol.
# Works in bull (breakouts) and bear (trends) via ADX filter.
name = "4h_Donchian20_1dADX_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        adx_val = adx_14_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with trend and volume
            if close_val > donch_high_val and adx_val > 25 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with trend and volume
            elif close_val < donch_low_val and adx_val > 25 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint
            if close_val <= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint
            if close_val >= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals