#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX trend strength (ADX > 25 = trending) and volume confirmation.
- Williams %R(14): Oversold < -80 for long, overbought > -20 for short.
- Entry: Long when Williams %R crosses above -80 AND ADX > 25 AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 AND ADX > 25 AND volume > 1.5 * 20-period average volume.
- Exit: Williams %R crosses above -50 for long exit, below -50 for short exit (mean reversion to midpoint).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in trending markets by buying dips in uptrends and selling rallies in downtrends.
- Avoids choppy markets (ADX <= 25) where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 30)  # Need 14 for Williams %R, 30 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Trend filter: only trade in trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: Williams %R crosses midpoint (-50)
        if position != 0:
            # Exit long: Williams %R crosses above -50
            if position == 1:
                if prev_williams_r < -50 and curr_williams_r >= -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses below -50
            elif position == -1:
                if prev_williams_r > -50 and curr_williams_r <= -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) in uptrend
            long_condition = (prev_williams_r < -80 and curr_williams_r >= -80 and 
                            trending and volume_confirm)
            
            # Short: Williams %R crosses below -20 (overbought reversal) in downtrend
            short_condition = (prev_williams_r > -20 and curr_williams_r <= -20 and 
                             trending and volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dADXTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0