#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and 4h volume confirmation.
Long when price breaks above Donchian upper band and 4h volume > 1.5x 20-period average and 1d ATR ratio < 0.8.
Short when price breaks below Donchian lower band and 4h volume > 1.5x 20-period average and 1d ATR ratio < 0.8.
Exit when price returns to Donchian midpoint or 1d ATR ratio > 1.2.
Uses 4h as primary timeframe, 4h for volume, 1d for volatility regime.
Designed to capture breakouts in low volatility environments which work in both bull and bear markets.
Target: 20-50 trades/year per symbol to minimize fee drag on 4h timeframe.
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
    open_ = prices['open'].values
    
    # Get 4h data for Donchian channels and volume filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        mid = (upper + lower) / 2
        return upper, lower, mid
    
    upper_20, lower_20, mid_20 = donchian_channels(high_4h, low_4h, 20)
    
    # Calculate 4h volume MA20
    volume_4h_series = pd.Series(volume_4h)
    vol_ma_20_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) and ATR ratio (current ATR / 20-period average ATR)
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_1d / atr_ma_20_1d, np.nan)
    
    # Align 4h Donchian channels, volume MA, and 1d ATR ratio to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_4h, mid_20)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Get aligned 4h volume for volume confirmation
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period average
        volume_confirmed = not np.isnan(volume_4h_aligned[i]) and \
                          not np.isnan(vol_ma_20_4h_aligned[i]) and \
                          volume_4h_aligned[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Volatility regime filter: 1d ATR ratio < 0.8 (low volatility environment)
        vol_regime = not np.isnan(atr_ratio_1d_aligned[i]) and atr_ratio_1d_aligned[i] < 0.8
        
        # Breakout conditions
        breakout_up = close[i] > upper_20_aligned[i]
        breakout_down = close[i] < lower_20_aligned[i]
        
        # Exit conditions: price returns to midpoint OR high volatility environment
        revert_to_mid = (position == 1 and close[i] < mid_20_aligned[i]) or \
                        (position == -1 and close[i] > mid_20_aligned[i])
        high_vol_exit = not np.isnan(atr_ratio_1d_aligned[i]) and atr_ratio_1d_aligned[i] > 1.2
        
        if position == 0:
            # Long: breakout above upper band with volume confirmation and low volatility regime
            if (breakout_up and volume_confirmed and vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band with volume confirmation and low volatility regime
            elif (breakout_down and volume_confirmed and vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR high volatility environment
            if (revert_to_mid or high_vol_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR high volatility environment
            if (revert_to_mid or high_vol_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_4hVolume_1dATRRegime_Breakout"
timeframe = "4h"
leverage = 1.0