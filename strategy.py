#!/usr/bin/env python3

# 4h_PreviousDayHighLow_Breakout
# Hypothesis: Breakout above previous day's high or below previous day's low with volume confirmation and ADX trend filter works in both bull and bear markets.
# Uses 4h timeframe to limit trade frequency (target: 20-50 trades/year). Previous day levels provide strong support/resistance.
# ADX filter ensures we only trade in trending conditions, reducing whipsaws in ranging markets.
# Volume confirmation ensures breakouts have conviction.

name = "4h_PreviousDayHighLow_Breakout"
timeframe = "4h"
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
    
    # Volume spike: >1.8x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for previous day high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day high/low (shifted by 1 to avoid look-ahead)
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    
    # Align previous day high/low to 4h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    
    # ADX(14) for trend filter on 4h timeframe
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initialize first values with simple average
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])
        dm_plus_smooth[13] = np.nanmean(dm_plus[1:15])
        dm_minus_smooth[13] = np.nanmean(dm_minus[1:15])
        
        # Wilder's smoothing for remaining values
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.full_like(atr, np.nan)
    di_minus = np.full_like(atr, np.nan)
    mask = ~np.isnan(atr) & (atr != 0)
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr[mask]
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan)
    dx_mask = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 27:  # Need 14 for DX + 13 more for ADX smoothing
        adx[26] = np.nanmean(dx[14:28])
        for i in range(27, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        if (np.isnan(prev_day_high_aligned[i]) or
            np.isnan(prev_day_low_aligned[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above previous day high + volume spike + ADX > 20
            if (close[i] > prev_day_high_aligned[i] and 
                volume_spike[i] and 
                adx[i] > 20):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below previous day low + volume spike + ADX > 20
            elif (close[i] < prev_day_low_aligned[i] and 
                  volume_spike[i] and 
                  adx[i] > 20):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below previous day high OR ADX drops below 15
            if (close[i] < prev_day_high_aligned[i]) or (adx[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above previous day low OR ADX drops below 15
            if (close[i] > prev_day_low_aligned[i]) or (adx[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals