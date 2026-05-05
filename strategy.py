#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter
# Long when: Price breaks above Donchian upper channel (20) AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending regime)
# Short when: Price breaks below Donchian lower channel (20) AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending regime)
# Exit when price returns to Donchian middle (mean of upper/lower) OR ADX < 20 (regime shift to range)
# Donchian breakout captures strong directional moves; volume spike confirms institutional interest; ADX filter ensures we only trade in trending markets
# Works in both bull and bear markets by trading breakouts in the direction of the trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_Donchian20_1dVolumeSpike_ADX_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX and volume average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range and ATR for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM for ADX
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR for ADX
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian Channel (20) on 12h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-period average
        # We need to get the current 1d volume - since we're on 12h timeframe,
        # we use the most recent completed 1d volume available
        vol_spike = volume_1d[-1] > 1.5 * vol_ma_20_1d_aligned[i] if len(volume_1d) > 0 else False
        
        # ADX regime: trending market (ADX > 25)
        trending_regime = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Break above upper Donchian in trending regime with volume spike
            if close[i] > donchian_upper[i] and trending_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian in trending regime with volume spike
            elif close[i] < donchian_lower[i] and trending_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle Donchian OR regime shift to range (ADX < 20)
            if close[i] < donchian_middle[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle Donchian OR regime shift to range (ADX < 20)
            if close[i] > donchian_middle[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals