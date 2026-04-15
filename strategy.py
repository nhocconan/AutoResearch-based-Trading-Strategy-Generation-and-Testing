#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h EMA crossover (21/55) with 1d ADX regime filter and volume confirmation.
# Uses 1d ADX(14) > 25 to identify trending markets (avoid chop) and 12h EMA21/EMA55 cross for entries.
# Volume filter ensures breakouts have sufficient momentum. Designed for low trade frequency
# (15-30/year) to minimize fee drag. Works in bull/bear: ADX filters non-trending, EMA crossover
# captures trend direction with confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 12h Indicators: EMA Crossover ===
    close_12h = pd.Series(df_12h['close'].values)
    ema_21_12h = close_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55_12h = close_12h.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_55_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_55_12h)
    
    # === 1d Indicators: ADX Trend Filter ===
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high_1d.diff()
    dm_minus = low_1d.diff() * -1  # inverted so down movement is positive
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed values
    tr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_14 = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_14 = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_14 / tr_14)
    di_minus = 100 * (dm_minus_14 / tr_14)
    
    # DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 12h volume > 1.3x 20-period 12h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_55_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. EMA21 > EMA55 (bullish crossover)
        # 2. 1d ADX > 25 (strong trending market)
        # 3. Volume confirmation
        if (ema_21_12h_aligned[i] > ema_55_12h_aligned[i] and
            adx_1d_aligned[i] > 25.0 and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. EMA21 < EMA55 (bearish crossover)
        # 2. 1d ADX > 25 (strong trending market)
        # 3. Volume confirmation
        elif (ema_21_12h_aligned[i] < ema_55_12h_aligned[i] and
              adx_1d_aligned[i] > 25.0 and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_EMA21_55_ADX1d_VolFilter_v1"
timeframe = "12h"
leverage = 1.0