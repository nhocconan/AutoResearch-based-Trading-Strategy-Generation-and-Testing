#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ATR breakout with 1d volume confirmation and 1w ADX trend filter
# ATR breakouts capture volatility expansions in trending markets. Volume confirms institutional participation.
# 1w ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# Exits when price reverts to ATR-based mean or trend weakens (ADX < 20).
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for strong trends only.

name = "1d_ATRBreakout_1dVolume_1wADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.full_like(close, np.nan)
    atr[13] = np.mean(tr[0:14])  # Simple average for first 14 periods
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder smoothing
    
    # ATR-based bands (mean ± 2*ATR)
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    upper_band = ma + 2 * atr
    lower_band = ma - 2 * atr
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 2.0)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr_w = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr_w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14_w = wilder_smooth(tr_w, 14)
    plus_dm14_w = wilder_smooth(plus_dm, 14)
    minus_dm14_w = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14_w != 0, 100 * (plus_dm14_w / tr14_w), 0)
    minus_di14 = np.where(tr14_w != 0, 100 * (minus_dm14_w / tr14_w), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx_w = wilder_smooth(dx, 14)
    
    adx_strong = adx_w > 25
    adx_weak = adx_w < 20
    adx_strong_aligned = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(ma[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, volume spike, strong trend
            if close[i] > upper_band[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, volume spike, strong trend
            elif close[i] < lower_band[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to mean or trend weakens
            if close[i] < ma[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to mean or trend weakens
            if close[i] > ma[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals