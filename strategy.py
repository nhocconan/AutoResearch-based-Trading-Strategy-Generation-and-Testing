#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime Filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power rising (improving) AND 1d ADX < 20 (range regime)
# Short when Bear Power > 0 AND Bull Power falling (worsening) AND 1d ADX < 20 (range regime)
# Exit when power diverges or ADX > 25 (trend regime)
# Elder Ray measures bull/bear strength relative to EMA13
# ADX regime filter ensures we trade mean reversion in low volatility environments
# Works in both bull and bear markets by fading overextended moves in ranging conditions
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.25

name = "6h_ElderRay_ADXRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Calculate 1d ADX for regime filtering
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Regime filters: ADX < 20 = range (mean revert), ADX > 25 = trend (avoid)
    adx_regime = adx < 20
    adx_trend = adx > 25
    
    # Align regime filters to 6h
    adx_regime_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_regime_aligned[i]) or np.isnan(adx_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime conditions
        in_range = bool(adx_regime_aligned[i])
        in_trend = bool(adx_trend_aligned[i])
        
        # Power momentum (1-bar change)
        bull_power_mom = bull_power[i] - bull_power[i-1] if i > 0 else 0
        bear_power_mom = bear_power[i] - bear_power[i-1] if i > 0 else 0
        
        if position == 0:
            # Long: Bull Power positive AND improving AND in range regime
            if bull_power[i] > 0 and bull_power_mom > 0 and in_range:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive AND improving AND in range regime
            elif bear_power[i] > 0 and bear_power_mom > 0 and in_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power deteriorating OR enters trend regime
            if bull_power_mom < 0 or in_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power deteriorating OR enters trend regime
            if bear_power_mom < 0 or in_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals