#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_Chop_Regime_v1
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction, confirmed by volume spike and choppiness regime filter.
Long when KAMA trending up + volume confirmation + chop regime allows trending; short when opposite.
Uses daily timeframe to minimize trades (target: 30-100 total over 4 years) and reduce fee impact.
Works in bull/bear by adapting to trend strength and filtering choppy markets where trend signals fail.
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
    
    # Calculate KAMA on primary timeframe (1d)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: slope over 5 periods
    kama_slope = np.diff(kama, n=5)
    kama_slope = np.concatenate([np.full(5, np.nan), kama_slope])
    kama_up = kama_slope > 0
    kama_down = kama_slope < 0
    
    # Load 1w HTF data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Choppiness Index (CHOP) over 14 periods on weekly
    atr_1w = np.zeros(len(close_1w))
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1])))
    atr_1w[1:] = tr_1w
    atr_1w = pd.Series(atr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = np.where(hh_14 != ll_14, 100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Regime: CHOP > 61.8 = ranging (avoid trend), CHOP < 38.2 = trending (allow trend)
    chop_ranging = chop_aligned > 61.8
    chop_trending = chop_aligned < 38.2
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(10, 14, 20) + 5  # KAMA seed + CHOP + volume EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: KAMA trending up + volume spike + chop regime allows trending
        if kama_up[i] and volume_spike[i] and chop_trending[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: KAMA trending down + volume spike + chop regime allows trending
        elif kama_down[i] and volume_spike[i] and chop_trending[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: loss of trend or chop becomes ranging
        elif position == 1 and (not kama_up[i] or chop_ranging[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not kama_down[i] or chop_ranging[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0