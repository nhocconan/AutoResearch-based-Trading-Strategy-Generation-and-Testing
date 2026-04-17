#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d volume spike filter and ADX trend confirmation.
Long when Williams %R < -80 (oversold) AND 1d volume > 2.0x 20-day average AND ADX > 25.
Short when Williams %R > -20 (overbought) AND 1d volume > 2.0x 20-day average AND ADX > 25.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ADX < 20.
Uses 4h for Williams %R calculation and 1d for volume and ADX filters to reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year). Williams %R captures momentum extremes,
volume confirmation ensures institutional participation, ADX filter avoids ranging markets.
Works in bull markets (buying oversold dips in uptrends) and bear markets (selling overbought rallies in downtrends).
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
    
    # Get 4h data for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Williams %R on 4h timeframe (14-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    close_4h_series = pd.Series(close_4h)
    
    # Highest high and lowest low over 14 periods
    highest_high = high_4h_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_4h_series.rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_4h) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ADX on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume 20-day moving average
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h Williams %R to 4h timeframe (no alignment needed)
    williams_r_aligned = williams_r
    
    # Align 1d ADX and volume MA to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol_1d = volume_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND volume > 2.0x avg AND ADX > 25 (trending)
            if wr < -80 and vol_1d > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume > 2.0x avg AND ADX > 25 (trending)
            elif wr > -20 and vol_1d > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (reversing up) OR ADX < 20 (range market)
            if wr > -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (reversing down) OR ADX < 20 (range market)
            if wr < -50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_VolumeSpike_ADX_Filter"
timeframe = "4h"
leverage = 1.0