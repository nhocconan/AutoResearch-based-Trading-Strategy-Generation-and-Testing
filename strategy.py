#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_1dTrend_RegimeFilter
Uses 4h Donchian channel breakout with 1d EMA50 trend filter and 1d chop regime.
Long when price breaks above Donchian(20) high in uptrend + low chop.
Short when price breaks below Donchian(20) low in downtrend + low chop.
Exit when price crosses midline or chop regime becomes choppy.
Designed for low trade frequency (~50-100 total trades over 4 years) to minimize fee drag.
Works in bull/bear markets by following 1d trend while using 4h Donchian breakouts for precise entries.
"""

name = "4h_Donchian_20_Breakout_1dTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 30-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d ATR for chop regime calculation (ATR(14))
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop regime: (ATR(14) / (max(high,20) - min(low,20))) * 100
    max_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    chop_denom = max_high_20 - min_low_20
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_value = (atr_14_1d / chop_denom) * 100
    chop_threshold_low = 38.2  # trending regime
    chop_threshold_high = 61.8  # ranging regime
    chop_regime = chop_value < chop_threshold_low  # True when trending (<38.2)
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align all indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(chop_regime_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + 1d EMA50 uptrend + trending chop regime + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                chop_regime_aligned[i] > 0.5 and  # trending regime
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + 1d EMA50 downtrend + trending chop regime + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  chop_regime_aligned[i] > 0.5 and  # trending regime
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below midline OR chop regime becomes choppy
            if (close[i] < donchian_mid[i]) or \
               (chop_regime_aligned[i] <= 0.5):  # choppy/ranging regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above midline OR chop regime becomes choppy
            if (close[i] > donchian_mid[i]) or \
               (chop_regime_aligned[i] <= 0.5):  # choppy/ranging regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals