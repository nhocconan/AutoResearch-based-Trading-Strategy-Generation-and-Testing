#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation.
- Williams %R(14) identifies overbought/oversold conditions: long when crosses above -80 from below, short when crosses below -20 from above
- 1d ADX(14) > 25 indicates trending market (use Williams signals), ADX < 20 indicates ranging market (fade Williams extremes)
- Volume confirmation: current volume > 1.5 * median volume of last 20 bars to avoid low-volatility false signals
- Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
- Works in both bull and bear markets: Williams %R captures mean reversion in ranges, ADX filters for trend strength
- Uses discrete position sizes (±0.25) to minimize fee churn
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
    
    # Williams %R(14): %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    plus_dm = np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])
    minus_dm = np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr = np.maximum(
        np.maximum(
            np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])),
            np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
        ),
        np.abs(df_1d['high'].values - df_1d['low'].values)
    )
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry logic
            if adx_aligned[i] > 25:  # Trending market
                # Long: Williams %R crosses above -80 from below (oversold bounce in uptrend)
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above (overbought rejection in downtrend)
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (ADX < 25)
                # Long: Williams %R crosses above -80 from below (oversold mean reversion)
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above (overbought mean reversion)
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) or ADX drops below 20 (trend weakening)
            if williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            elif adx_aligned[i] < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) or ADX drops below 20 (trend weakening)
            if williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            elif adx_aligned[i] < 20:  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0