#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_ATRFilter
Hypothesis: Trade Camarilla pivot R1/S1 breakouts on 1h with volume and ATR filters.
Use 4h trend (EMA50) and 1d regime (ADX) for signal direction, 1h only for entry timing.
Targets 15-30 trades/year per symbol. Works in bull/bear: 4h/1d filters avoid counter-trend trades.
Position size: 0.20.
"""

name = "1h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema50_4h[i] = multiplier * close_4h[i] + (1 - multiplier) * ema50_4h[i-1]
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d ADX for regime filter (trending > 25)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        # Initial ATR
        atr[13] = np.nanmean(tr[1:15])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Smooth DM+
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    if len(dm_plus) >= 14:
        dm_plus_smooth[13] = np.nansum(dm_plus[1:15])
        dm_minus_smooth[13] = np.nansum(dm_minus[1:15])
        for i in range(14, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.full_like(dm_plus, np.nan)
    di_minus = np.full_like(dm_minus, np.nan)
    valid = (~np.isnan(atr)) & (atr != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan)
    di_sum = di_plus + di_minus
    valid_dx = (~np.isnan(di_sum)) & (di_sum != 0)
    dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]
    
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 14:
        # Initial ADX
        adx[27] = np.nanmean(dx[14:28])
        for i in range(28, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h ATR for volatility filter and stop
    tr_1h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_1h[0] = high[0] - low[0]
    atr_1h = np.full_like(tr_1h, np.nan)
    if len(tr_1h) >= 14:
        atr_1h[13] = np.mean(tr_1h[1:15])
        for i in range(14, len(tr_1h)):
            atr_1h[i] = (atr_1h[i-1] * 13 + tr_1h[i]) / 14
    
    # Calculate Camarilla levels for previous day
    # Using previous day's high, low, close
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla multipliers
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 1h
    R1_1h = align_htf_to_ltf(prices, df_1d, R1)
    S1_1h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(R1_1h[i]) or np.isnan(S1_1h[i]) or np.isnan(atr_1h[i])):
            signals[i] = 0.0
            continue
        
        # Apply filters
        if not (session_filter[i] and volume_filter[i]):
            signals[i] = 0.0
            continue
        
        # Determine regime: ADX > 25 = trending
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 in uptrend (4h EMA50 up) OR in ranging market
            if close[i] > R1_1h[i] and (close_4h_aligned[i] > ema50_4h_aligned[i] or not is_trending):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 in downtrend (4h EMA50 down) OR in ranging market
            elif close[i] < S1_1h[i] and (close_4h_aligned[i] < ema50_4h_aligned[i] or not is_trending):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 OR trend reversal
            if close[i] < S1_1h[i] or (is_trending and close_4h_aligned[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price above R1 OR trend reversal
            if close[i] > R1_1h[i] or (is_trending and close_4h_aligned[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals