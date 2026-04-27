#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + 1d EMA trend filter + volume spike.
# Choppiness Index (14) > 61.8 indicates ranging market (mean-reversion opportunity).
# In ranging market: long when price < BB lower band (20,2), short when price > BB upper band.
# Trending market (CHOP < 38.2): follow 1d EMA trend (long if price > EMA, short if price < EMA).
# Volume spike (>1.5x 20-period average) confirms participation.
# Designed for ~20-30 trades/year per symbol, works in both bull and bear via regime adaptation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr1 = np.zeros(n)
    tr1[0] = high[0] - low[0]
    for i in range(1, n):
        tr1[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = np.zeros(n)
    for i in range(14, n):
        atr14[i] = np.mean(tr1[i-13:i+1])
    
    sum_tr14 = np.zeros(n)
    for i in range(14, n):
        sum_tr14[i] = np.sum(tr1[i-13:i+1])
    
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    chop = np.full(n, 50.0)
    for i in range(14, n):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(sum_tr14[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Bollinger Bands (20,2)
    sma20 = np.zeros(n)
    std20 = np.zeros(n)
    for i in range(20, n):
        sma20[i] = np.mean(close[i-19:i+1])
        std20[i] = np.std(close[i-19:i+1])
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if is_ranging:
            # Mean reversion in ranging market
            if close[i] < bb_lower[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] > bb_upper[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position or flat
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        elif is_trending:
            # Follow trend in trending market
            if close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position or flat
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Neutral chop zone - stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_ChoppinessIndex_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0