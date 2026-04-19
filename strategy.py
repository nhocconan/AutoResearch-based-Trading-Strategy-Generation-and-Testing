#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour Williams %R for momentum reversal signals, filtered by 1-day ADX trend strength and volume confirmation.
# Williams %R identifies overbought/oversold conditions; ADX ensures we only trade in trending markets (ADX > 25).
# Volume confirmation filters out low-conviction moves. Designed to capture reversals in both bull and bear markets.
# Target: 20-40 trades/year per symbol with disciplined entries to minimize fee drag.
name = "4h_WilliamsR_ADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(w):
        result = np.zeros_like(w)
        result[13] = np.sum(w[:14])  # First 14-period sum
        for i in range(14, len(w)):
            result[i] = result[i-1] - (result[i-1]/14) + w[i]
        return result
    
    tr14 = smooth(tr)
    dm_plus_14 = smooth(dm_plus)
    dm_minus_14 = smooth(dm_minus)
    
    # Directional Indicators
    plus_di = 100 * dm_plus_14 / tr14
    minus_di = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = np.zeros_like(tr14)
    mask = (plus_di + minus_di) != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = smooth(dx)
    adx_14 = adx  # Already smoothed
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 12-hour Williams %R for momentum reversal
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R calculation
    highest_high = np.maximum.accumulate(high_12h)
    lowest_low = np.minimum.accumulate(low_12h)
    
    # For proper lookback, we need rolling window
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    highest_high_14 = rolling_max(high_12h, 14)
    lowest_low_14 = rolling_min(low_12h, 14)
    
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # Avoid division by zero
    
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX indicates trending market (> 25)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # Long: Williams %R oversold (< -80) with volume spike
                if (williams_r_12h_aligned[i] < -80 and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20) with volume spike
                elif (williams_r_12h_aligned[i] > -20 and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                    
            elif position == 1:
                # Long: exit if Williams %R rises above -50 (momentum fading) or ADX weakens
                if (williams_r_12h_aligned[i] > -50) or (adx_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
            elif position == -1:
                # Short: exit if Williams %R falls below -50 (momentum fading) or ADX weakens
                if (williams_r_12h_aligned[i] < -50) or (adx_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets, stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals