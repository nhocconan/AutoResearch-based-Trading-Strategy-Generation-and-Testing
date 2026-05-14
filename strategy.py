#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Extreme with 1d ADX trend filter and 6h volume spike (>2.5x 20-period average).
# Williams %R measures overbought/oversold levels. Long when %R < -80 (oversold) AND 1d ADX > 25 (strong trend) AND volume spike.
# Short when %R > -20 (overbought) AND 1d ADX > 25 (strong trend) AND volume spike.
# Exit when %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (weak trend).
# Uses 1d HTF for trend strength to avoid whipsaws in ranging markets. Volume spike filters for institutional participation.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag for 6h timeframe.
# Williams %R is effective in both bull and bear markets when combined with trend confirmation, as it identifies exhaustion points.

name = "6h_WilliamsR_Extreme_1dADX25_6hVolumeSpike_v1"
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
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Replace division by zero or near-zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h volume spike: > 2.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - trend strength filter
    # True Range
    tr1 = pd.Series(high_1d - low_1d).abs()
    tr2 = pd.Series(high_1d - pd.Series(close_1d).shift(1)).abs()
    tr3 = pd.Series(low_1d - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d - pd.Series(high_1d).shift(1)).clip(lower=0).values
    dm_minus = pd.Series(pd.Series(low_1d).shift(1) - low_1d).clip(lower=0).values
    dm_plus = np.where(tr == 0, 0, dm_plus)
    dm_minus = np.where(tr == 0, 0, dm_minus)
    
    # Smoothed DM and TR
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / atr_14
    di_minus = 100 * dm_minus_14 / atr_14
    # Avoid division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (waits for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume spike
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] > 25 and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (strong trend) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (recovering from oversold) OR ADX < 20 (weak trend)
            if (williams_r[i] > -50 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (declining from overbought) OR ADX < 20 (weak trend)
            if (williams_r[i] < -50 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals