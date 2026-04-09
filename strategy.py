#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppy regime filter.
# Donchian(20) breakouts capture trends, volume > 1.5x 20-period average confirms strength,
# and Choppiness Index > 61.8 ensures we only trade in ranging markets (mean reversion bias).
# Works in bull/bear by fading false breakouts in chop and catching real breakouts in trends.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (log10(highest_high - lowest_low) * n)) / log10(n)
    tr1 = pd.Series(high).rolling(2).apply(lambda x: x[0] - x[1]).abs().values  # high - prev_close
    tr2 = pd.Series(low).rolling(2).apply(lambda x: x[0] - x[1]).abs().values   # low - prev_close
    tr3 = (high - low).values
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high_14 - lowest_low_14
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values /
                          (np.log10(hl_range) * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8 = ranging)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel OR chop drops below 38.2 (trending)
            if close[i] < donchian_upper[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel OR chop drops below 38.2 (trending)
            if close[i] > donchian_lower[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_filter:
                # Long entry: price breaks above Donchian upper band
                if close[i] > donchian_upper[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower band
                elif close[i] < donchian_lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals