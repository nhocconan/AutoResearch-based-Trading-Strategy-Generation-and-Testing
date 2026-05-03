#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly ATR regime filter and volume confirmation
# Donchian breakouts capture momentum in both bull and bear markets.
# Weekly ATR regime filter: only trade when weekly ATR(14) is below its 50-period EMA (low volatility regime) to avoid whipsaws.
# Volume confirmation ensures breakouts have participation. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via upward breaks and bear markets via downward breaks, but only in low-volatility regimes where trends are cleaner.

name = "6h_Donchian20_WeeklyATRRegime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w ATR(14) and its 50-period EMA for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # ATR(14)
    atr_14_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(tr)):
        atr_14_1w[i] = np.nanmean(tr[i-13:i+1])
    
    # EMA50 of ATR
    atr_series = pd.Series(atr_14_1w)
    atr_ema_50_1w = atr_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Regime: low volatility when ATR(14) < EMA50(ATR)
    atr_regime_low = atr_14_1w < atr_ema_50_1w
    atr_regime_aligned = align_htf_to_ltf(prices, df_1w, atr_regime_low.astype(float))
    
    # Calculate Donchian channels from previous 6h bar (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(atr_regime_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        atr_regime = bool(atr_regime_aligned[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian in low volatility regime with volume spike
            if close[i] > donchian_upper[i] and atr_regime and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian in low volatility regime with volume spike
            elif close[i] < donchian_lower[i] and atr_regime and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or regime changes to high volatility
            if close[i] < donchian_lower[i] or not atr_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or regime changes to high volatility
            if close[i] > donchian_upper[i] or not atr_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals