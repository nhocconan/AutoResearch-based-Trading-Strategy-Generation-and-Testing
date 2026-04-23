#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band breakout with 1d ADX trend filter and volume confirmation
- Uses 6h Bollinger Bands (20, 2.0) for breakout signals
- 1d ADX(14) > 25 defines trending regime (avoid whipsaws in ranging markets)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading breakouts only in trending regimes
- ADX regime filter prevents entries during sideways consolidation
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
    
    # Calculate 6h Bollinger Bands (20, 2.0)
    period = 20
    std_dev = 2.0
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    # Calculate 1d ADX(14) for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed averages
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Align indicators to 6h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)  # Using 1d index for alignment
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Bollinger Bands, ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_upper = close[i] > upper_band_aligned[i]
        price_below_lower = close[i] < lower_band_aligned[i]
        
        # Trending regime filter
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above upper BB, trending regime, volume spike
            long_signal = (price_above_upper and 
                          trending and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below lower BB, trending regime, volume spike
            short_signal = (price_below_lower and 
                           trending and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Bollinger Band breakout or trend weakening
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower BB or trend weakens
                if (price_below_lower or 
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper BB or trend weakens
                if (price_above_upper or 
                    adx_aligned[i] < 20):  # Trend weakening
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_BB20_Breakout_1dADXTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0