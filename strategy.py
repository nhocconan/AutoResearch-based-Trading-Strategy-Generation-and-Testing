#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d ADX(14) for trend strength (ADX > 25 = trending market).
- Williams %R(14): Long when %R < -80 (oversold) in uptrend, Short when %R > -20 (overbought) in downtrend.
- Volume: Current 4h volume > 1.5 * 20-period volume MA to confirm participation.
- Entry: Long when Williams %R < -80 AND 1d ADX > 25 AND 1d close > 1d EMA20 (uptrend) AND volume spike.
         Short when Williams %R > -20 AND 1d ADX > 25 AND 1d close < 1d EMA20 (downtrend) AND volume spike.
- Exit: Williams %R reverts to -50 (mean reversion) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe.
This strategy captures mean reversion moves within strong trends, avoiding choppy markets.
Williams %R identifies exhaustion points, ADX ensures we only trade in trending conditions,
and volume confirmation filters false signals. Works in both bull and bear markets by
trading pullbacks in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend direction
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[1:])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]),
                       np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)
    dm_minus = np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]),
                        np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 1d
    highest_high = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d_close) / (highest_high - lowest_low)
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for ADX and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend
            if volume_spike[i] and adx_aligned[i] > 25:
                # Bullish: Williams %R oversold (< -80) AND uptrend (close > EMA20)
                if williams_r_aligned[i] < -80 and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R overbought (> -20) AND downtrend (close < EMA20)
                elif williams_r_aligned[i] > -20 and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reverts to -50 OR loss of volume confirmation OR trend weakens
            if williams_r_aligned[i] > -50 or not volume_spike[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts to -50 OR loss of volume confirmation OR trend weakens
            if williams_r_aligned[i] < -50 or not volume_spike[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dADX_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0