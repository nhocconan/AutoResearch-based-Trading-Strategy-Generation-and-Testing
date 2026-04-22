#!/usr/bin/env python3

"""
Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) upper band with ADX>25 and volume spike;
short when price breaks below Donchian(20) lower band with ADX>25 and volume spike.
Uses ADX to filter ranging markets and only trade in trending conditions. Designed
for moderate trade frequency (20-50 trades/year) by requiring breakout, trend, and volume.
Works in both bull and bear markets by following the trend direction from higher timeframe.
"""

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
    
    # Donchian Channel (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ADX for trend strength - load ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily data
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
    
    # Smoothed values
    def smoothed_series(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) 
        # Subsequent values smoothed
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = smoothed_series(tr, 14)
    dm_plus_smooth = smoothed_series(dm_plus, 14)
    dm_minus_smooth = smoothed_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_series(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donch_high[i-1]  # Using previous close to avoid lookahead
        breakout_down = close[i] < donch_low[i-1]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: breakout up + strong trend + volume spike
            if breakout_up and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + strong trend + volume spike
            elif breakout_down and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: breakout in opposite direction or trend weakening
            exit_signal = False
            
            if position == 1:
                # Exit long: breakout down or ADX drops below 20
                if breakout_down or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: breakout up or ADX drops below 20
                if breakout_up or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_AdxTrend_Volume"
timeframe = "4h"
leverage = 1.0