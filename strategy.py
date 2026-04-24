#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX Trend Filter and Volume Confirmation.
- Long when price breaks above upper Bollinger Band AND 1d ADX > 25 (trending regime) AND volume > 1.5 * 20-period average
- Short when price breaks below lower Bollinger Band AND 1d ADX > 25 (trending regime) AND volume > 1.5 * 20-period average
- Bollinger Band squeeze (bandwidth < 20th percentile) precedes breakout for higher probability
- Exit when price returns to middle Bollinger Band (20-period SMA)
- Uses 6h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Bollinger Bands provide volatility-based structure; ADX filters for trending markets; volume confirms breakout strength
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets by filtering only on ADX > 25
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Bollinger Bands (20, 2) on 6h data
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Calculate Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / basis
    # Squeeze condition: bandwidth below 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Trend filter: trending if ADX > 25
    trending_regime = adx_aligned > 25
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14+14, 20)  # Bollinger Bands, squeeze percentile, ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Bollinger Band AND trending regime AND volume confirmation AND squeeze condition
            if close[i] > upper_band[i] and trending_regime[i] and volume_confirm[i] and squeeze_condition[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Bollinger Band AND trending regime AND volume confirmation AND squeeze condition
            elif close[i] < lower_band[i] and trending_regime[i] and volume_confirm[i] and squeeze_condition[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: return to middle Bollinger Band (basis)
            if close[i] >= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: return to middle Bollinger Band (basis)
            if close[i] <= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_ADXTrend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0