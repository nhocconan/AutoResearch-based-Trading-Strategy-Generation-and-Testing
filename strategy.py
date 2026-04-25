#!/usr/bin/env python3
"""
1d_Trend_Following_With_Choppiness_Filter_v1
Hypothesis: Ride 1d trends using EMA20/50 crossovers filtered by 1w Choppiness Index to avoid whipsaws in ranging markets.
Only trade when EMA20 > EMA50 for longs (or EMA20 < EMA50 for shorts) AND 1w market is trending (CHOP < 50).
Exit on opposite EMA crossover or when market becomes choppy (CHOP >= 60).
Position size: 0.30 to balance capture and fee drag.
Target: 10-20 trades/year to stay well under 150-trade 1d hard max.
Works in bull (captures uptrends) and bear (captures downtrends) markets by being directionally flexible.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 and EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for Choppiness Index regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Choppiness Index (CHOP) for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    chop_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high_1w).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low_1w).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    # Choppiness Index: 100 * log10(sum(ATR) / (highest_high - lowest_low)) / log10(chop_period)
    chop = 100 * np.log10(pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values / hl_range) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to neutral if not enough data
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and CHOP (14)
    start_idx = max(50, chop_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Determine 1d HTF trend (bullish = EMA20 > EMA50)
        htf_1d_bullish = ema_20_1d_aligned[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = ema_20_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 50), exit when choppy (CHOP >= 60)
        is_trending = chop_aligned[i] < 50.0
        is_choppy = chop_aligned[i] >= 60.0
        
        if position == 0:
            # Long setup: EMA20 > EMA50 + trending regime
            long_setup = htf_1d_bullish and is_trending
            
            # Short setup: EMA20 < EMA50 + trending regime
            short_setup = htf_1d_bearish and is_trending
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: EMA20 <= EMA50 (trend reversal) OR market becomes choppy
            if (not htf_1d_bullish) or is_choppy:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: EMA20 >= EMA50 (trend reversal) OR market becomes choppy
            if (htf_1d_bullish) or is_choppy:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Trend_Following_With_Choppiness_Filter_v1"
timeframe = "1d"
leverage = 1.0