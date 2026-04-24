#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d ADX regime filter + volume confirmation.
- Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5 * median volume
- Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5 * median volume
- Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (range regime)
- Uses 6h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Williams %R identifies extreme pullbacks in trends; ADX ensures we only trade strong trends
- Volume confirmation avoids low-momentum false signals
- Designed for BTC/ETH: works in bull trends (buy pullbacks) and bear trends (sell rallies)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    up_move[up_move < 0] = 0
    down_move[down_move < 0] = 0
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(up_move.values).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(down_move.values).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di_14 = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), ADX > 25 (trending), volume confirmation
            if williams_r_aligned[i] < -80 and adx_14_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), ADX > 25 (trending), volume confirmation
            elif williams_r_aligned[i] > -20 and adx_14_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR ADX < 20 (range regime)
            if williams_r_aligned[i] > -50 or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR ADX < 20 (range regime)
            if williams_r_aligned[i] < -50 or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0