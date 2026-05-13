#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above R3 (Camarilla resistance 3) AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending market)
# Short when price breaks below S3 (Camarilla support 3) AND 1d volume > 2.0x 20-period average AND 1d ADX > 25
# Exit when price returns to Camarilla pivot point (PP) OR ADX < 20 (range market)
# Uses 6h timeframe for lower frequency, Camarilla levels from 1d for structure, volume spike for confirmation, ADX for regime.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "6h_Camarilla_R3S3_Breakout_1dVolume_ADX_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/2
    # S3 = PP - (H - L) * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Volume filter: current 1d volume > 2.0x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # ADX calculation (14-period)
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # +DM = max(H - H_prev, 0) if > max(L_prev - L, 0) else 0
    # -DM = max(L_prev - L, 0) if > max(H - H_prev, 0) else 0
    high_diff = np.diff(high_1d, prepend=np.nan)
    low_diff = -np.diff(low_1d, prepend=np.nan)  # negative of diff so positive when low decreases
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # +DI = 100 * smoothed +DM / smoothed TR
    # -DI = 100 * smoothed -DM / smoothed TR
    plus_di_1d = 100.0 * plus_dm_smooth / tr_smooth
    minus_di_1d = 100.0 * minus_dm_smooth / tr_smooth
    
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    dx_1d = 100.0 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    # ADX = smoothed DX
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND volume spike AND ADX > 25 (trending)
            if close[i] > r3_1d_aligned[i] and volume_filter_1d_aligned[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND volume spike AND ADX > 25 (trending)
            elif close[i] < s3_1d_aligned[i] and volume_filter_1d_aligned[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point OR ADX < 20 (range market)
            if close[i] <= pp_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point OR ADX < 20 (range market)
            if close[i] >= pp_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals