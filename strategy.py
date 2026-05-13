#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d ADX regime filter and volume confirmation.
# In bear markets (ADX > 25), extreme Williams %R readings (< -80 for oversold, > -20 for overbought) 
# often precede mean-reversion bounces. Volume spike confirms institutional participation.
# Uses discrete sizing (0.25) to limit drawdown. Targets 12-30 trades/year by requiring 
# confluence of extreme momentum, trend strength, and volume. Works in both bull and bear 
# markets: ADX regime filter avoids false signals in ranging markets, while Williams %R 
# captures exhaustion moves that often reverse sharply.

name = "6h_WilliamsR_Reversal_1dADX_Volume"
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
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation (14-period)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback_wr = 14
    highest_high = pd.Series(high_6h).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low_6h).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = ((highest_high - close_6h) / (highest_high - lowest_low)) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    lookback_adx = 14
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) in trending market (ADX > 25) with volume spike
            if williams_r_aligned[i] < -80 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) in trending market (ADX > 25) with volume spike
            elif williams_r_aligned[i] > -20 and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (momentum weakening)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 (momentum weakening)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals