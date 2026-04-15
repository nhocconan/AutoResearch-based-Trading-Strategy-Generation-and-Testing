#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND 1d +DI > -DI
# Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND 1d -DI > +DI
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending markets only.
# This combination avoids whipsaws in ranging markets and captures strong trends in both bull/bear regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX, +DI, -DI (using Wilder's smoothing) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    di_plus_1d = 100 * wilders_smoothing(dm_plus, period) / atr_1d
    di_minus_1d = 100 * wilders_smoothing(dm_minus, period) / atr_1d
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # Align 1d indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    di_plus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_plus_1d)
    di_minus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_minus_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 30) + 5  # EMA13 + ADX(14,14) + buffer
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(di_plus_1d_aligned[i]) or np.isnan(di_minus_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (price above EMA13 on upside)
        # 2. Bear Power < 0 (price above EMA13 on downside)
        # 3. 1d ADX > 25 (strong trend)
        # 4. 1d +DI > -DI (bullish directional bias)
        if (bull_power[i] > 0) and \
           (bear_power[i] < 0) and \
           (adx_1d_aligned[i] > 25) and \
           (di_plus_1d_aligned[i] > di_minus_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (price below EMA13 on downside)
        # 2. Bull Power > 0 (price below EMA13 on upside)
        # 3. 1d ADX > 25 (strong trend)
        # 4. 1d -DI > +DI (bearish directional bias)
        elif (bear_power[i] < 0) and \
             (bull_power[i] > 0) and \
             (adx_1d_aligned[i] > 25) and \
             (di_minus_1d_aligned[i] > di_plus_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dADX_Directional_Filter_v1"
timeframe = "6h"
leverage = 1.0