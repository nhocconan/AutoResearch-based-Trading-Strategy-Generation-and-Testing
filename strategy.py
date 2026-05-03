#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR-based volatility regime filter + volume confirmation
# Donchian breakouts capture strong momentum moves in both bull and bear markets.
# 1d ATR(14) / SMA(50) ratio identifies low-volatility regimes where breakouts are more reliable.
# Volume confirmation (1.5x 20-period EMA) filters false breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

name = "12h_Donchian20_1dATRRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ATR-based regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and SMA(50) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    
    # SMA(50) of close
    close_series = pd.Series(close_1d)
    sma_50 = close_series.rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: ATR(14) / SMA(50) - identifies low volatility regimes
    atr_ratio = np.where(sma_50 != 0, atr_14 / sma_50, np.nan)
    
    # Low volatility regime: ATR ratio < 0.02 (adjustable threshold)
    low_vol_regime = atr_ratio < 0.02
    
    # Align 1d low volatility regime to 12h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    # Calculate Donchian channels from previous 12h bar (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA on 12h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(low_vol_regime_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Low volatility regime condition
        in_low_vol = low_vol_regime_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian in low vol regime with volume spike
            if close[i] > donchian_upper[i] and in_low_vol and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian in low vol regime with volume spike
            elif close[i] < donchian_lower[i] and in_low_vol and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or leaves low vol regime
            if close[i] < donchian_lower[i] or not in_low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or leaves low vol regime
            if close[i] > donchian_upper[i] or not in_low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals