#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter
Hypothesis: 12-hour Camarilla R1/S1 breakout with daily EMA34 trend filter and choppiness regime filter.
Enters long when price breaks above R1 with bullish daily trend and choppy market (mean reversion).
Enters short when price breaks below S1 with bearish daily trend and choppy market.
Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed for 50-150 total trades over 4 years.
Chop filter reduces whipsaw in trending markets, improving performance in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) on daily timeframe
    # Use prior completed daily bar to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC for Camarilla calculation (shifted by 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate pivot point and Camarilla levels
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load daily data for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index on 12h timeframe (regime filter)
    # CHOP > 61.8 = ranging market (good for mean reversion at pivot levels)
    # CHOP < 38.2 = trending market (avoid false breakouts)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_sum = max_high - min_low
    range_sum = np.where(range_sum == 0, 1e-10, range_sum)
    
    chop = 100 * np.log10(atr * atr_period / range_sum) / np.log10(atr_period)
    chopping_market = chop > 61.8  # ranging/choppy market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 1-day shift + 34-day EMA + ATR period)
    start_idx = max(1 + 34, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chopping_market[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 + bullish daily trend + choppy market
        if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and chopping_market[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S1 + bearish daily trend + choppy market
        elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and chopping_market[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s1_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_aligned[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter"
timeframe = "12h"
leverage = 1.0