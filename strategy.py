#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h for balanced trade frequency and noise reduction.
- HTF: 1d ADX(14) for trend strength (ADX > 25 = trending market).
- Williams %R(14): Extreme readings below -80 (oversold) or above -20 (overbought).
- Volume: Current 6h volume > 1.8 * 20-period 6h volume MA to confirm participation.
- Entry: Long when %R < -80 AND ADX > 25 AND volume spike.
         Short when %R > -20 AND ADX > 25 AND volume spike.
- Exit: Opposite %R level (%R > -20 for long, %R < -80 for short) or ADX < 20.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 80-180 total trades over 4 years (20-45/year) for 6h timeframe.
This strategy captures mean reversion extremes in trending markets, avoiding choppy conditions
where Williams %R fails. Volume spikes confirm institutional interest at turning points.
Works in both bull and bear markets by only trading strong trends (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    denom = highest_high - lowest_low
    wpct = np.where(denom != 0, -100 * (highest_high - close) / denom, -50.0)
    
    # Get 1d data for ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high - df_1d_low)
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = tr1[0]  # first bar
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low),
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0.0), 0.0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)),
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0.0), 0.0)
    dm_plus[0] = 0.0
    dm_minus[0] = 0.0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/14)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0.0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0.0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0.0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period 6h volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA
    volume_spike = volume > (1.8 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need enough bars for volume MA and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(wpct[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_wpct = wpct[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and ADX > 25
            if volume_spike[i] and adx_val > 25.0:
                # Bullish: Williams %R < -80 (oversold)
                if curr_wpct < -80.0:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R > -20 (overbought)
                elif curr_wpct > -20.0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (overbought) OR ADX < 20 (weakening trend)
            if curr_wpct > -20.0 or adx_val < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (oversold) OR ADX < 20 (weakening trend)
            if curr_wpct < -80.0 or adx_val < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0