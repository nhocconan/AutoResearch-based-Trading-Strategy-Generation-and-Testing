#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and 1D ADX Trend Filter
Long: Price breaks above 1D Donchian(20) high + volume > 2.0x 12h volume MA + 1D ADX(14) > 25
Short: Price breaks below 1D Donchian(20) low + volume > 2.0x 12h volume MA + 1D ADX(14) > 25
Exit: Opposite break of 1D Donchian level
Uses 1D ADX to filter for trending markets, reducing false breakouts in chop
Target: 15-25 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # 1D Donchian channels (20-period)
    donch_high = df_1d['high'].rolling(window=20, min_periods=20).max()
    donch_low = df_1d['low'].rolling(window=20, min_periods=20).min()
    
    # 1D ADX(14) for trend strength
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean()
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean()
    
    # DI values
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Align all 1D indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # 12h volume moving average (20-period for confirmation)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_20 = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_12h = align_htf_to_ltf(prices, df_12h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_12h[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: break above 1D Donchian high + volume spike + trending market
            if price > donch_high_aligned[i] and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below 1D Donchian low + volume spike + trending market
            elif price < donch_low_aligned[i] and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below 1D Donchian low
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 1D Donchian high
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ADX25"
timeframe = "12h"
leverage = 1.0