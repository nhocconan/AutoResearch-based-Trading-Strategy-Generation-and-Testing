#!/usr/bin/env python3
"""
4h_RollingDonchianBreakout_VolumeFilter_ADXTrend
Hypothesis: Use rolling Donchian(20) breakouts on 4h timeframe with volume confirmation (>1.5x 20-period average) and ADX(14) trend filter (>25). Long when price breaks above upper band with volume and ADX>25, short when breaks below lower band with volume and ADX>25. Exit when price crosses the midline (average of upper/lower bands). Designed to capture trends in both bull and bear markets with volume confirmation to avoid false breakouts. Target ~30-50 trades/year to minimize fee drag.
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
    
    # Rolling Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ADX(14) for trend filter
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(20, 14, 14)  # Donchian(20), volume avg(20), ADX(14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation AND ADX>25 (trending)
            if close[i] > high_max[i] and volume_confirm[i] and adx[i] > 25:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian band with volume confirmation AND ADX>25 (trending)
            elif close[i] < low_min[i] and volume_confirm[i] and adx[i] > 25:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below midline (mean reversion) or trend weakens
            if close[i] < donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above midline (mean reversion) or trend weakens
            if close[i] > donchian_mid[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RollingDonchianBreakout_VolumeFilter_ADXTrend"
timeframe = "4h"
leverage = 1.0