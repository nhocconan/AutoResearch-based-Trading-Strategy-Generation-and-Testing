#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation.
# Uses Donchian channel breakouts for structure, 1d ADX > 25 to ensure trending market (avoids chop),
# and volume > 1.5x 20-bar SMA to confirm momentum. Designed for low trade frequency (20-40/year)
# to minimize fee drag. Works in bull/bear: ADX filter ensures we only trade strong trends,
# Donchian breakouts capture momentum, volume confirmation avoids false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_high_20 = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_4h.rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # === 1d Indicators: ADX(14) for trend strength ===
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d - high_1d.shift(1)
    down_move = low_1d.shift(1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 4h volume > 1.5x 20-period 4h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period)
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and
            adx_1d_aligned[i] > 25 and
            vol_confirm):
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period)
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and
              adx_1d_aligned[i] > 25 and
              vol_confirm):
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_ADX1d_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0