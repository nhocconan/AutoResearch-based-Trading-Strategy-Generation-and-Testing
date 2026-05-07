#!/usr/bin/env python3
# 4h_ADX_Trend_With_Volume_Confirmation
# Hypothesis: Trend-following strategy using ADX(14) from daily timeframe to identify strong trends,
# combined with 4h EMA50 for direction and volume confirmation to reduce false signals.
# Designed to work in both bull and bear markets by only taking trades when trend strength (ADX > 25)
# is present. Targets 20-40 trades/year to avoid excessive fee churn.

name = "4h_ADX_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement (+DM and -DM)
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, +DM, -DM over 14 periods (Wilder's smoothing)
    n_1d = len(high_1d)
    atr = np.zeros(n_1d)
    dm_plus_smooth = np.zeros(n_1d)
    dm_minus_smooth = np.zeros(n_1d)
    
    # Initial values (simple average of first 14 periods)
    if n_1d >= 14:
        atr[13] = np.sum(tr[:14])
        dm_plus_smooth[13] = np.sum(dm_plus[:14])
        dm_minus_smooth[13] = np.sum(dm_minus[:14])
        
        # Wilder's smoothing for subsequent periods
        for i in range(14, n_1d):
            atr[i] = atr[i-1] - (atr[i-1] / 14) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / 14) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / 14) + dm_minus[i]
    
    # Calculate Directional Indicators (+DI and -DI)
    plus_di = np.zeros(n_1d)
    minus_di = np.zeros(n_1d)
    for i in range(14, n_1d):
        if atr[i] > 0:
            plus_di[i] = 100 * (dm_plus_smooth[i] / atr[i])
            minus_di[i] = 100 * (dm_minus_smooth[i] / atr[i])
    
    # Calculate DX and ADX
    dx = np.zeros(n_1d)
    for i in range(14, n_1d):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n_1d)
    if n_1d >= 28:  # Need 14 periods for DX smoothing
        adx[27] = np.sum(dx[14:28])  # First ADX is average of first 14 DX values
        for i in range(28, n_1d):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14  # Wilder's smoothing
    
    # Align ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(adx_4h[i]) or np.isnan(ema_50_4h[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter only when trend is strong (ADX > 25) and volume confirms
            if adx_4h[i] > 25 and volume[i] > vol_ma[i]:
                # Long: strong trend + price above EMA50
                if close[i] > ema_50_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: strong trend + price below EMA50
                elif close[i] < ema_50_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: trend weakens or price crosses below EMA50
            if adx_4h[i] < 25 or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakens or price crosses above EMA50
            if adx_4h[i] < 25 or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals