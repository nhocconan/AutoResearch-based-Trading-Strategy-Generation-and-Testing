#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. ADX > 25 filters for trending markets
# where mean reversion works best (pullbacks to the mean). Volume > 1.2x average confirms
# institutional participation. Discrete sizing 0.25. Target: 50-150 total trades over 4 years.
# Works in bull markets via long pullbacks in uptrends and in bear markets via short bounces
# in downtrends, avoiding choppy regimes where mean reversion fails.

name = "6h_WilliamsR_MeanReversion_1dADXTrend_VolumeConfirm_v1"
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
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    # Handle division by zero
    plus_di_1d = np.where(atr_1d == 0, 0, plus_di_1d)
    minus_di_1d = np.where(atr_1d == 0, 0, minus_di_1d)
    
    # DX and ADX
    dx_denom = plus_di_1d + minus_di_1d
    dx = np.where(dx_denom != 0, 100 * np.abs(plus_di_1d - minus_di_1d) / dx_denom, 0)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # LONG: Williams %R oversold (< -80) AND ADX > 25 (trending) AND volume > 1.2x average
        if (williams_r[i] < -80 and 
            adx_1d_aligned[i] > 25 and 
            volume[i] > 1.2 * avg_volume[i]):
            signals[i] = 0.25
        # SHORT: Williams %R overbought (> -20) AND ADX > 25 (trending) AND volume > 1.2x average
        elif (williams_r[i] > -20 and 
              adx_1d_aligned[i] > 25 and 
              volume[i] > 1.2 * avg_volume[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals