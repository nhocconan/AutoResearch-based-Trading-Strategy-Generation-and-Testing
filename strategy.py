#!/usr/bin/env python3
# Hypothesis: 4h Williams %R Extreme with 1d ADX25 trend filter and 4h volume confirmation (>1.5x 20-period average).
# Williams %R > -20 = overbought (short signal), < -80 = oversold (long signal).
# Only trade in direction of 1d ADX > 25 (strong trend) to avoid whipsaws in ranging markets.
# Volume confirmation >1.5x 20-period average ensures institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe to stay within fee drag limits.
# Williams %R identifies exhaustion points; ADX filters for trending environments where reversals are meaningful.

name = "4h_WilliamsR_Extreme_1dADX25_4hVolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # 4h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - trend strength filter
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff().values
    dm_minus = -pd.Series(low_1d).diff().values
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    # Smoothed DM and ATR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R lookback
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume confirm
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND volume confirm
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (return from oversold) OR ADX < 20 (trend weakening)
            if (williams_r[i] > -50 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (return from overbought) OR ADX < 20 (trend weakening)
            if (williams_r[i] < -50 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals