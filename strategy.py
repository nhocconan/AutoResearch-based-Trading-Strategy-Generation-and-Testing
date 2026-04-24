#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX regime and Elder Ray calculation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d EMA13).
- ADX regime: ADX > 25 = trending (trade with momentum), ADX < 20 = ranging (fade extremes).
- Entry Logic:
    * Trending regime (ADX > 25): Long when Bull Power > 0 and volume spike, Short when Bear Power < 0 and volume spike.
    * Ranging regime (ADX < 20): Long when Bear Power < -threshold and volume spike (oversold bounce), 
                                 Short when Bull Power > threshold and volume spike (overbought fade).
- Exit: Reverse signal or when power returns to zero (mean reversion in ranging, momentum exhaustion in trending).
- Volume confirmation: 6h volume > 1.5 * 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying strength in uptrends, in bear via selling weakness in downtrends, and fading extremes in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_elder_ray(high, low, close, ema_len=13):
    """Calculate Elder Ray Bull Power and Bear Power for given OHLC and EMA"""
    if len(close) < ema_len:
        return np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    ema = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full_like(close, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA13 and ADX14
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = df_1d['high'].values - ema_13
    bear_power_1d = df_1d['low'].values - ema_13
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Align 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)  # For reference
    
    # Volume confirmation: 6h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Threshold for fading signals in ranging market
    fade_threshold = 0.5  # Will be adjusted dynamically based on typical power values
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Dynamic fade threshold based on recent power volatility
        if i >= 50:
            recent_bull = bull_power_aligned[max(0, i-50):i]
            recent_bear = bear_power_aligned[max(0, i-50):i]
            valid_bull = recent_bull[~np.isnan(recent_bull)]
            valid_bear = recent_bear[~np.isnan(recent_bear)]
            if len(valid_bull) > 10 and len(valid_bear) > 10:
                fade_threshold = 0.5 * (np.nanstd(valid_bull) + np.nanstd(valid_bear))
        
        if position == 0:
            if adx_aligned[i] > 25:  # Trending regime
                # Trade with momentum: long on bullish power, short on bearish power
                if bull_power_aligned[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power_aligned[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:  # Ranging regime
                # Fade extremes: long on oversold, short on overbought
                if bear_power_aligned[i] < -fade_threshold and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif bull_power_aligned[i] > fade_threshold and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: power fades or reverses
            if adx_aligned[i] > 25:  # In trend, exit when power weakens
                if bull_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # In range, exit when power returns to zero (mean reversion)
                if bull_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: power fades or reverses
            if adx_aligned[i] > 25:  # In trend, exit when power weakens
                if bear_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # In range, exit when power returns to zero (mean reversion)
                if bear_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0