#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band squeeze breakout with 12-hour volume confirmation and daily trend filter.
Enters long when price breaks above upper Bollinger Band during low volatility (squeeze) with above-average 12h volume and daily uptrend.
Enters short when price breaks below lower Bollinger Band during squeeze with above-average 12h volume and daily downtrend.
Uses Bollinger Bands (20,2) to identify volatility contractions and expansions, which often precede strong moves.
The squeeze acts as a volatility filter to avoid whipsaws in ranging markets, while breakouts capture trending moves.
Works in both bull and bear markets by following the daily trend direction.
Target: 20-30 trades/year per symbol to minimize fee drag.
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
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20,2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Bollinger Band width for squeeze detection (normalized by basis)
    bb_width = (upper - lower) / basis
    # Squeeze threshold: BB width below its 50-period mean (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_mean = bb_width_series.rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_mean
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate daily close for trend filter
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB (20) and BB width mean (50) -> max 50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(squeeze[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_1d = close_1d_aligned[i]
        
        # Bollinger Band values
        upper_now = upper[i]
        lower_now = lower[i]
        squeeze_now = squeeze[i]
        
        # Volume filter: volume > 1.2x 12h average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Entry conditions: BB breakout during squeeze with volume + daily trend
        if position == 0:
            # Long: price breaks above upper BB during squeeze with volume + daily uptrend
            if price_now > upper_now and squeeze_now and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price breaks below lower BB during squeeze with volume + daily downtrend
            elif price_now < lower_now and squeeze_now and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below basis (mean reversion) or volatility expands
            if price_now < basis[i] or not squeeze_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above basis or volatility expands
            if price_now > basis[i] or not squeeze_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_BollingerSqueezeBreakout_12hVolume_1dTrend"
timeframe = "4h"
leverage = 1.0