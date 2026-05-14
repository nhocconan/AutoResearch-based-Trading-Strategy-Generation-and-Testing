#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX regime filter and 6h volume confirmation (>1.5x 20-period average).
# Williams %R identifies overbought/oversold conditions. In strong trends (ADX>25), extreme %R readings often precede continuations rather than reversals.
# Long: %R < -80 (oversold) AND ADX > 25 (strong trend) AND close > EMA20 (trend alignment) AND volume > 1.5x MA20.
# Short: %R > -20 (overbought) AND ADX > 25 (strong trend) AND close < EMA20 (trend alignment) AND volume > 1.5x MA20.
# Exit: %R crosses back above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weakness).
# Uses 1d HTF for ADX regime to avoid false signals in ranging markets. Volume confirmation filters low-quality breakouts.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity with fee drag for 6h timeframe.

name = "6h_WilliamsR_Extreme_1dADX_Regime_6hVolumeConfirm_v1"
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
    # Williams %R (14 period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # EMA20 for trend alignment
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX (14 period) - trend strength filter
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    # Smoothed DM and ATR
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    atr_14_smooth = pd.Series(atr_14).rolling(window=14, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / atr_14_smooth
    di_minus = 100 * dm_minus_14 / atr_14_smooth
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(volume_confirm_6h[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND close > EMA20 AND volume confirm
            if (williams_r[i] < -80 and 
                adx_14_aligned[i] > 25 and 
                close[i] > ema_20[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND close < EMA20 AND volume confirm
            elif (williams_r[i] > -20 and 
                  adx_14_aligned[i] > 25 and 
                  close[i] < ema_20[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (recovering from oversold) OR ADX < 20 (trend weakness)
            if (williams_r[i] > -50 or 
                adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (declining from overbought) OR ADX < 20 (trend weakness)
            if (williams_r[i] < -50 or 
                adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals