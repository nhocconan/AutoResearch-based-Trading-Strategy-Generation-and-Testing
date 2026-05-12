#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_And_Chop
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h timeframe to detect trend,
combined with volume confirmation and Choppiness Index regime filter to avoid whipsaws.
Long: KAMA rising + volume > 1.5x SMA20 + Choppiness > 50 (trending regime)
Short: KAMA falling + volume > 1.5x SMA20 + Choppiness > 50
Exit: Opposite KAMA direction or Choppiness < 40 (range regime)
Targets 12-37 trades/year to stay within fee limits. Uses 1w trend filter for robustness.
"""

name = "12h_KAMA_Trend_With_Volume_And_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.concatenate([[0], volatility[1:]])
    
    er = np.zeros_like(close)
    er[er_period:] = change[er_period:] / np.maximum(volatility[er_period:], 1e-10)
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama_out = np.zeros_like(close)
    kama_out[0] = close[0]
    
    for i in range(1, len(close)):
        kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
    
    return kama_out

def choppiness_index(high, low, close, period=14):
    """Choppiness Index: higher = ranging, lower = trending"""
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    for i in range(1, len(close)):
        atr[i] = np.mean(tr[max(0, i-period+1):i+1])
    
    sum_atr = np.zeros_like(close)
    for i in range(period-1, len(close)):
        sum_atr[i] = np.sum(atr[i-period+1:i+1])
    
    max_min_range = np.zeros_like(close)
    for i in range(period-1, len(close)):
        max_min_range[i] = np.max(high[i-period+1:i+1]) - np.min(low[i-period+1:i+1])
    
    cpi = np.zeros_like(close)
    for i in range(period-1, len(close)):
        if max_min_range[i] > 0:
            cpi[i] = 100 * np.log10(sum_atr[i] / max_min_range[i]) / np.log10(period)
        else:
            cpi[i] = 50
    
    return cpi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate KAMA on 12h data
    kama_val = kama(close, er_period=10, fast=2, slow=30)
    
    # Calculate Choppiness Index
    chop_val = choppiness_index(high, low, close, period=14)
    
    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Get aligned values
        ema50_aligned = ema50_1w_aligned[i]
        chop_val_i = chop_val[i]
        vol_threshold_val = volume_threshold[i]
        kama_current = kama_val[i]
        kama_prev = kama_val[i-1]

        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned) or np.isnan(chop_val_i) or 
            np.isnan(vol_threshold_val) or np.isnan(kama_current) or np.isnan(kama_prev)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        kama_rising = kama_current > kama_prev
        kama_falling = kama_current < kama_prev
        trending_regime = chop_val_i > 50  # Chop > 50 indicates trending
        ranging_regime = chop_val_i < 40   # Chop < 40 indicates ranging

        if position == 0:
            # LONG: KAMA rising + volume spike + trending regime + above weekly EMA50
            if (kama_rising and
                volume[i] > vol_threshold_val and
                trending_regime and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling + volume spike + trending regime + below weekly EMA50
            elif (kama_falling and
                  volume[i] > vol_threshold_val and
                  trending_regime and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR ranging regime
            if (kama_falling or ranging_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR ranging regime
            if (kama_rising or ranging_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals