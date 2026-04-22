#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R reversal with 1-day ADX trend filter and volume confirmation.
Long when %R crosses above -80 from oversold with 1-day ADX > 25 and volume spike.
Short when %R crosses below -20 from overbought with 1-day ADX > 25 and volume spike.
Exit when %R crosses -50 (centerline) in opposite direction.
Williams %R captures momentum reversals; ADX filters for trending markets; volume confirms strength.
Designed for low trade frequency by requiring multiple confirmations. Works in both bull and bear
markets by trading reversals within the prevailing trend.
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
    
    # Load 1-day data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) 
        # Subsequent values: smoothed = previous * (period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R (14) on 4h data
    def williams_r(high, low, close, period):
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        # For proper lookback, we need to use rolling window
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (hh - close) / (hh - ll)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr[i]) or np.isnan(wr[i-1]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: %R crosses above -80 from oversold with ADX > 25 and volume spike
            if (wr[i] > -80 and wr[i-1] <= -80 and 
                adx_1d_aligned[i] > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: %R crosses below -20 from overbought with ADX > 25 and volume spike
            elif (wr[i] < -20 and wr[i-1] >= -20 and 
                  adx_1d_aligned[i] > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: %R crosses -50 (centerline) in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: %R crosses below -50
                if wr[i] < -50 and wr[i-1] >= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: %R crosses above -50
                if wr[i] > -50 and wr[i-1] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0