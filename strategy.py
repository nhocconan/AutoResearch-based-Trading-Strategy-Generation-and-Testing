#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging/mean reversion, outside for trending). Long when price breaks above DC(20) upper band with volume confirmation in trending market (CHOP < 38.2); short when price breaks below DC(20) lower band with volume confirmation in trending market (CHOP < 38.2). In ranging market (CHOP > 61.8), mean reversion: long when price touches DC(20) lower band, short when touches upper band. Uses 12h HTF for trend confirmation via HMA(21) alignment. Discrete position sizing 0.25. Designed for low turnover (target: 20-50 trades/year) by requiring confluence of breakout, volume, and regime. Works in both bull (trend following) and bear (mean reversion in ranging markets).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_choppiness(high, low, close, window=14):
    """Calculate Choppiness Index (CHOP)"""
    atr_sum = pd.Series(high - low).rolling(window=window, min_periods=window).sum()
    highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
    lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
    return chop.values

def calculate_hma(values, window):
    """Calculate Hull Moving Average"""
    n = len(values)
    if n < window:
        return np.full(n, np.nan)
    half = window // 2
    sqrt = int(np.sqrt(window))
    wma2 = pd.Series(values).rolling(window=half, min_periods=half).mean()
    wma1 = pd.Series(values).rolling(window=window, min_periods=window).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).rolling(window=sqrt, min_periods=sqrt).mean()
    return hma.values

name = "4h_donchian_breakout_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian Channel (20)
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # 12h HTF HMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(chop[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filters
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        is_transition = 38.2 <= chop[i] <= 61.8  # neutral zone, no trades
        
        if position == 1:  # Long position
            # Exit conditions
            if is_trending and close[i] < hma_12h_aligned[i]:  # trend reversal
                position = 0
                signals[i] = 0.0
            elif is_ranging and close[i] >= (dc_upper[i] + dc_lower[i]) / 2:  # mean reversion to midpoint
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending and close[i] > hma_12h_aligned[i]:  # trend reversal
                position = 0
                signals[i] = 0.0
            elif is_ranging and close[i] <= (dc_upper[i] + dc_lower[i]) / 2:  # mean reversion to midpoint
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if is_transition or not volume_confirmed:
                signals[i] = 0.0
                continue
                
            # Enter based on regime
            if is_trending:
                # Trend following: breakout in direction of HTF trend
                if close[i] > dc_upper[i] and close[i] > hma_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < dc_lower[i] and close[i] < hma_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif is_ranging:
                # Mean reversion: touch bands
                if close[i] <= dc_lower[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= dc_upper[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals