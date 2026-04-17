#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d ATR volatility regime filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 4h Donchian upper band in low volatility regime (ATR ratio < 0.8) with volume > 1.3x 20-period average.
Short when price breaks below 4h Donchian lower band in low volatility regime with volume > 1.3x 20-period average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Low volatility breakouts capture explosive moves after consolidation; volume confirms institutional participation.
Designed to work in bull markets (breakout continuation) and bear markets (volatility expansion after panic).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 1d ATR (14-period) for volatility regime
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # first value is simple average
            result[period-1] = np.nanmean(data[:period])
            # subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    
    # Calculate 1d ATR 50-period average for regime classification
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period average ATR)
    # ATR ratio < 0.8 = low volatility regime (consolidation)
    # ATR ratio > 1.2 = high volatility regime (expansion)
    atr_ratio = atr_14 / atr_ma_50
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high_4h, low_4h, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ATR MA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR ratio < 0.8 (consolidation phase)
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume_4h_aligned[i] > 1.3 * vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper band in low vol with volume
            if (close[i] > donchian_upper_aligned[i] and 
                low_vol_regime and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian lower band in low vol with volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  low_vol_regime and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian middle (mean reversion)
            donchian_middle = (donchian_upper_4h + donchian_lower_4h) / 2
            donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian middle
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dATR_VolRegime_4hDonchian20_Volume_Confirm"
timeframe = "6h"
leverage = 1.0