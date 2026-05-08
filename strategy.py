#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above Donchian upper band AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Short when price breaks below Donchian lower band AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Uses ATR(14) for dynamic stoploss (exit when price moves against position by 2.5x ATR).
# Target: 20-50 trades per year to minimize fee drag while capturing strong trends.
# Works in bull markets (trend following) and avoids chop via ADX filter.

name = "4h_Donchian_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian(20) on 4h
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    for i in range(20, len(high)):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # ATR(14) for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.zeros_like(close)
    atr[14] = np.mean(tr[:15])
    for i in range(15, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-20:i])
    
    # 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_series(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    tr_smoothed = smooth_series(tr_1d, 14)
    dm_plus_smoothed = smooth_series(dm_plus, 14)
    dm_minus_smoothed = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_series(dx, 14)
    
    # Align 1d indicators to 4h
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 15, 20, 14+13, 14+13)  # Donchian, ATR, vol MA, ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-period average
        # Find corresponding 1d bar index for current 4h bar
        vol_spike = False
        if i >= 16:  # Need at least one full 1d bar (16x4h bars)
            # Approximate: use the most recent completed 1d bar
            vol_1d_idx = min(len(vol_1d)-1, i // 16)
            if vol_1d_idx < len(vol_1d) and vol_1d_idx >= 0:
                vol_spike = vol_1d[vol_1d_idx] > 1.5 * vol_ma_20[vol_1d_idx]
        
        if position == 0:
            # Long entry: Donchian breakout up + volume spike + ADX > 25
            if close[i] > donch_high[i] and vol_spike and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakout down + volume spike + ADX > 25
            elif close[i] < donch_low[i] and vol_spike and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR adverse move of 2.5x ATR
            if close[i] < donch_low[i] or close[i] < (prices['close'].iloc[i-1] - 2.5 * atr[i]) if i > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout OR adverse move of 2.5x ATR
            if close[i] > donch_high[i] or close[i] > (prices['close'].iloc[i-1] + 2.5 * atr[i]) if i > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals