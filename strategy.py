#!/usr/bin/env python3
# 6h_elder_ray_regime_v1
# Hypothesis: 6h strategy using Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d regime filter (ADX > 25 = trending, ADX < 20 = ranging). In trending regimes: enter long when Bull Power > 0 and rising, short when Bear Power > 0 and rising. In ranging regimes: fade extreme Elder Ray values (long when Bull Power < -std, short when Bear Power < -std). Uses discrete position sizing (0.25) to limit fee drag. Designed for 12-37 trades/year to work in both bull and bear markets by adapting to volatility regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
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
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

name = "6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray and trend
    ema13 = calculate_ema(close, 13)
    
    # Elder Ray Index
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # 1d HTF regime filter: ADX(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    
    adx_14_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Rising Bull/Bear Power (1-bar momentum)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # Volatility regime for ranging market (std of Bull Power over 20 periods)
    bull_power_std = pd.Series(bull_power).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(bull_power_std[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative or ADX drops below 20 (regime change to ranging)
            if bull_power[i] <= 0 or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns negative or ADX drops below 20 (regime change to ranging)
            if bear_power[i] <= 0 or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if adx > 25:  # Trending regime
                # Enter long: Bull Power positive and rising
                if bull_power[i] > 0 and bull_power_rising[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: Bear Power positive and rising
                elif bear_power[i] > 0 and bear_power_rising[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime (ADX < 20)
                # Fade extreme Elder Ray values
                if bull_power[i] < -bull_power_std[i]:  # Extremely bearish, mean revert long
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] < -bull_power_std[i]:  # Extremely bullish, mean revert short
                    position = -1
                    signals[i] = -0.25
    
    return signals