#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold), 1d ADX < 25 (range/weak trend), and volume > 1.3x 20-period average.
# Short when Williams %R > -20 (overbought), 1d ADX < 25, and volume > 1.3x 20-period average.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Williams %R effective in ranging markets; ADX filter avoids strong trends where mean reversion fails.

name = "6h_WilliamsR_MeanReversion_1dADX_Regime_Volume_v1"
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
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold + weak trend (range) + volume confirmation
            if williams_r[i] < -80 and adx_aligned[i] < 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought + weak trend (range) + volume confirmation
            elif williams_r[i] > -20 and adx_aligned[i] < 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (mean reversion complete) OR ADX strengthens
            if williams_r[i] > -50 or adx_aligned[i] >= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 OR ADX strengthens
            if williams_r[i] < -50 or adx_aligned[i] >= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals