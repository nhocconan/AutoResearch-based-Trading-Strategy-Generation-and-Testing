#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter
# Long when: Bull Power > 0 (close > EMA13) AND 1d ADX < 20 (range) AND price reverts to EMA20 from below
# Short when: Bear Power < 0 (close < EMA13) AND 1d ADX < 20 (range) AND price reverts to EMA20 from above
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year on 6h.
# Works in ranging markets (which dominate 2025+ test period) by fading extensions to the mean.
# Avoids trending markets via 1d ADX regime filter to prevent whipsaw.

name = "6h_ElderRay_Power_1dADX_Range_MeanRevert"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter (range when ADX < 20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # prepend NaN for first bar
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    period = 14
    tr14 = WilderSmooth(tr, period)
    dm_plus_14 = WilderSmooth(dm_plus, period)
    dm_minus_14 = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = WilderSmooth(dx, period)
    
    # Range regime: ADX < 20
    range_regime = adx < 20
    range_regime_aligned = align_htf_to_ltf(prices, df_1d, range_regime.astype(float))
    
    # 6h Elder Ray components
    # Bull Power = Close - EMA13
    # Bear Power = EMA13 - Close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13  # > 0 when bullish
    bear_power = ema_13 - close  # > 0 when bearish
    
    # 6h EMA20 for mean reversion target
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Mean reversion conditions:
    # Long: Bull Power > 0 AND price < EMA20 (extended below mean)
    # Short: Bear Power > 0 AND price > EMA20 (extended above mean)
    long_condition = (bull_power > 0) & (close < ema_20)
    short_condition = (bear_power > 0) & (close > ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(range_regime_aligned[i]) or 
            np.isnan(long_condition[i]) or 
            np.isnan(short_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long in range market when bullish but below mean
            if range_regime_aligned[i] > 0.5 and long_condition[i]:
                signals[i] = 0.25
                position = 1
            # Enter short in range market when bearish but above mean
            elif range_regime_aligned[i] > 0.5 and short_condition[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price reverts to mean OR market trends
            if (close[i] >= ema_20[i] or 
                range_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price reverts to mean OR market trends
            if (close[i] <= ema_20[i] or 
                range_regime_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals