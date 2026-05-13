#!/usr/bin/env python3
"""
1d_MultiFactor_Positive_Momentum_With_Regime_Filter
Hypothesis: Combine positive momentum (price above 100-day EMA) with volume strength (volume above 50-day average) and regime filter (price below 200-day EMA to avoid extended trends) for long entries. Short when price below 50-day EMA, volume strong, and price above 200-day EMA. Uses daily timeframe to limit trades and avoid fee drift. Designed to work in both bull and bear markets by adapting to trend strength via EMA filters.
"""

name = "1d_MultiFactor_Positive_Momentum_With_Regime_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate multiple EMAs for trend and regime filters
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100 = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate volume average (50-day) for volume strength filter
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_100_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume strength condition: current volume > 1.3x 50-day average
        vol_strong = volume[i] > 1.3 * vol_ma_50[i]
        
        if position == 0:
            # LONG: Price above 100-day EMA (uptrend), volume strong, and price below 200-day EMA (not overextended)
            if close[i] > ema_100_aligned[i] and vol_strong and close[i] < ema_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below 50-day EMA (downtrend), volume strong, and price above 200-day EMA (not overextended down)
            elif close[i] < ema_50_aligned[i] and vol_strong and close[i] > ema_200_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 100-day EMA or volume weakens significantly
            if close[i] < ema_100_aligned[i] or volume[i] < 0.7 * vol_ma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 50-day EMA or volume weakens significantly
            if close[i] > ema_50_aligned[i] or volume[i] < 0.7 * vol_ma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals