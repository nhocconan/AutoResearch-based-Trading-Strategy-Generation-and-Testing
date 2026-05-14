#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extremes with 1d ADX25 trend filter and 6h volume spike (>2.0x 20-period average).
# Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup).
# Long when Williams %R crosses above -80 from below AND 1d ADX > 25 (strong trend) AND volume > 2.0x MA20.
# Short when Williams %R crosses below -20 from above AND 1d ADX > 25 (strong trend) AND volume > 2.0x MA20.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weakening).
# Uses 1d HTF for trend strength to reduce noise and overtrading. Volume confirmation (>2.0x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.
# Williams %R is a momentum oscillator effective in both bull and bear markets when combined with trend filter.

name = "6h_WilliamsR_Extreme_1dADX25_6hVolumeConfirm_v1"
timeframe = "6h"
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
    
    # --- 6h Indicators (LTF) ---
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    # 6h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (2.0 * vol_ma_20)
    
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
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff().values
    dm_minus = -pd.Series(low_1d).diff().values
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND ADX > 25 AND volume confirm
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                adx_aligned[i] > 25 and
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above AND ADX > 25 AND volume confirm
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  adx_aligned[i] > 25 and
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 OR ADX < 20 (trend weakening)
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
               adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 OR ADX < 20 (trend weakening)
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or \
               adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals