#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume spike filter and ADX trend confirmation
# Long when: price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period MA AND 1d ADX > 25
# Short when: price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period MA AND 1d ADX > 25
# Exit when: price crosses Donchian midpoint OR ADX drops below 20
# Uses Donchian for structure, volume for conviction, ADX for trend filter
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_DonchianBreakout_1dVolumeSpike_ADXFilter"
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
    
    # Calculate Donchian Channel on 4h (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate volume spike filter on 1d
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(vol_1d), dtype=bool)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    else:
        adx = np.full(len(high_1d), np.nan)
    
    # ADX trend strength filters
    adx_strong = np.zeros(len(adx), dtype=bool)
    adx_weak = np.zeros(len(adx), dtype=bool)
    for i in range(len(adx)):
        if not np.isnan(adx[i]):
            adx_strong[i] = adx[i] > 25
            adx_weak[i] = adx[i] < 20
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_strong_aligned[i]) or np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high + volume spike + strong ADX
            if (close[i] > donchian_high[i] and 
                volume_spike_aligned[i] == 1.0 and 
                adx_strong_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + volume spike + strong ADX
            elif (close[i] < donchian_low[i] and 
                  volume_spike_aligned[i] == 1.0 and 
                  adx_strong_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR weak ADX
            if (close[i] < donchian_mid[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR weak ADX
            if (close[i] > donchian_mid[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals