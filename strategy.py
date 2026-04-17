#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend_v1
Breakout above/below Donchian(20) with volume confirmation and ADX(14) trend filter.
Exit when price returns to Donchian middle or opposite band.
Uses 1d ADX for higher timeframe trend alignment.
Designed to capture sustained moves with volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === Volume MA (20) for confirmation ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ADX(14) for trend strength ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === 1d ADX for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    tr1d = df_1d['high'] - df_1d['low']
    tr2d = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3d = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr_d = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    tr_d[0] = tr1d[0]
    
    plus_dm_d = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), 
                         np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm_d = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), 
                          np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    plus_dm_d = np.concatenate([[0], plus_dm_d])
    minus_dm_d = np.concatenate([[0], minus_dm_d])
    
    atr_d = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_d = 100 * pd.Series(plus_dm_d).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_d + 1e-10)
    minus_di_d = 100 * pd.Series(minus_dm_d).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_d + 1e-10)
    dx_d = 100 * np.abs(plus_di_d - minus_di_d) / (plus_di_d + minus_di_d + 1e-10)
    adx_1d = pd.Series(dx_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper band, volume > MA, ADX > 20, 1d ADX > 20
            if (close[i] > donchian_high[i] and 
                volume[i] > volume_ma[i] and 
                adx[i] > 20 and 
                adx_1d_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower band, volume > MA, ADX > 20, 1d ADX > 20
            elif (close[i] < donchian_low[i] and 
                  volume[i] > volume_ma[i] and 
                  adx[i] > 20 and 
                  adx_1d_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to middle line or opposite band
        elif position == 1:
            # Exit long: price crosses below middle or above upper (failed breakout)
            if close[i] < donchian_mid[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle or below lower (failed breakdown)
            if close[i] > donchian_mid[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0