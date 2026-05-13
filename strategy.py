#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold), 1d ADX < 25 (range market), and volume > 1.5x average.
# Short when Williams %R > -20 (overbought), 1d ADX < 25 (range market), and volume > 1.5x average.
# Exit when Williams %R crosses -50 (mean reversion completion) or volume drops below average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Williams %R identifies extreme price levels; 1d ADX ensures we only trade in ranging markets where mean reversion works; volume confirms the reversal attempt.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets (range-bound conditions occur in all regimes).

name = "6h_WilliamsR_MeanReversion_1dADX25_Regime_VolumeConfirm"
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
    
    # Calculate 1d ADX for regime filter (trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R (14-period)
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_wr, lookback_vol, 14) + 1, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold (%R < -80), ranging market (ADX < 25), volume spike
            if (williams_r[i] < -80 and 
                adx_1d_aligned[i] < 25 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought (%R > -20), ranging market (ADX < 25), volume spike
            elif (williams_r[i] > -20 and 
                  adx_1d_aligned[i] < 25 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion complete (%R crosses -50) OR volume drops
            if (williams_r[i] > -50 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion complete (%R crosses -50) OR volume drops
            if (williams_r[i] < -50 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals