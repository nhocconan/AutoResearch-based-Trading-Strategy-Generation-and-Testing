#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d ADX Regime + Volume Spike
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Williams %R calculation.
- Williams %R(14) from 1d: Extreme oversold (< -90) for long, extreme overbought (> -10) for short.
- Regime filter: 1d ADX(14) > 25 for trending markets (follow momentum), ADX < 20 for ranging (mean revert).
- In trending regime (ADX > 25): Extreme Williams %R signals continuation (breakout).
- In ranging regime (ADX < 20): Extreme Williams %R signals mean reversion (fade).
- Volume confirmation: current 6h volume > 1.5x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via capturing momentum extremes, in bear via fading extremes in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R and ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ADX(14) on 1d for regime filter
    # True Range
    tr1 = high_1d - low_1d
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Williams %R extremes: oversold < -90, overbought > -10
    williams_oversold = williams_r < -90
    williams_overbought = williams_r > -10
    
    # ADX regime: trending > 25, ranging < 20
    adx_trending = adx > 25
    adx_ranging = adx < 20
    
    # Align HTF indicators to 6h
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Williams %R/ADX + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or
            np.isnan(adx_trending_aligned[i]) or np.isnan(adx_ranging_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Convert aligned arrays to boolean for logic
        is_oversold = bool(williams_oversold_aligned[i])
        is_overbought = bool(williams_overbought_aligned[i])
        is_trending = bool(adx_trending_aligned[i])
        is_ranging = bool(adx_ranging_aligned[i])
        has_volume = bool(volume_spike[i])
        
        if position == 0:
            # Entry logic based on regime
            if is_trending and has_volume:
                # Trending regime: follow momentum from extremes
                if is_oversold:
                    # Extreme oversold in uptrend -> long (breakout continuation)
                    signals[i] = 0.25
                    position = 1
                elif is_overbought:
                    # Extreme overbought in downtrend -> short (breakout continuation)
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and has_volume:
                # Ranging regime: fade extremes (mean reversion)
                if is_oversold:
                    # Extreme oversold in range -> long (mean reversion bounce)
                    signals[i] = 0.25
                    position = 1
                elif is_overbought:
                    # Extreme overbought in range -> short (mean reversion fade)
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: opposite extreme or loss of volume/momentum
            if is_overbought or not has_volume:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite extreme or loss of volume/momentum
            if is_oversold or not has_volume:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0