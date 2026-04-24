#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation.
- Williams %R(14): long when crosses above -80 from below (oversold bounce),
  short when crosses below -20 from above (overbought rejection)
- 1d ADX(14) > 25: trending regime (take signals in direction of 1d EMA50)
- 1d ADX(14) <= 25: ranging regime (fade extremes: long at %R < -80, short at %R > -20)
- Volume must be > 1.5 * median volume of last 20 bars (avoid low-volume fakeouts)
- Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
- Williams %R captures mean reversion in ranging markets and momentum in trending markets
- ADX regime filter adapts strategy to market conditions, reducing whipsaws
- Works in both bull and bear markets by switching between trend following and mean reversion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r[highest_high == lowest_low] = -50
    
    # Get 1d data ONCE before loop for ADX and EMA50 regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr[0] = high_1d[0] - low_1d[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus[(dm_plus < 0) | (dm_plus < dm_minus)] = 0
    dm_minus[(dm_minus < 0) | (dm_minus < dm_plus)] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[di_plus + di_minus == 0] = 0
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime: trending (ADX > 25) or ranging (ADX <= 25)
        is_trending = adx_14_aligned[i] > 25
        
        if position == 0:
            if is_trending:
                # Trending regime: follow 1d EMA50 direction
                # Long: Williams %R crosses above -80 from below AND price > 1d EMA50
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above AND price < 1d EMA50
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging regime: fade extremes
                # Long: Williams %R below -80 (oversold) AND volume confirmation
                if williams_r[i] < -80 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R above -20 (overbought) AND volume confirmation
                elif williams_r[i] > -20 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR trend reversal in ranging regime
            if williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR trend reversal in ranging regime
            if williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0