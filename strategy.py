#!/usr/bin/env python3
"""
4h_Squeeze_Breakout_Pullback_v1
Hypothesis: Combines Bollinger Band squeeze detection with Donchian breakout and pullback entries.
In low volatility (BB width < 20th percentile), wait for Donchian(20) breakout, then enter on pullback to 20 EMA.
Works in both bull/bear by capturing volatility expansion after consolidation. Low frequency due to squeeze filter.
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
    
    # Bollinger Bands (20, 2)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_width = (upper - lower) / ma20
    
    # Squeeze: BB width below 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_percentile
    
    # Donchian channels (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA20 for pullback
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Wait for indicators to warm up
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ma20[i]) or np.isnan(donch_high[i]) or np.isnan(ema20[i]) or np.isnan(vol_avg[i]):
            continue
            
        if position == 0:
            # Look for squeeze breakout with pullback entry
            if squeeze[i-1]:  # Was in squeeze previous bar
                # Bullish breakout: price breaks above Donchian high
                if close[i] > donch_high[i]:
                    # Wait for pullback to EMA20 (but not below)
                    if low[i] <= ema20[i] and high[i] >= ema20[i] and vol_filter[i]:
                        signals[i] = size
                        position = 1
                # Bearish breakout: price breaks below Donchian low
                elif close[i] < donch_low[i]:
                    # Wait for pullback to EMA20 (but not above)
                    if high[i] >= ema20[i] and low[i] <= ema20[i] and vol_filter[i]:
                        signals[i] = -size
                        position = -1
        elif position == 1:
            # Exit long: price closes below EMA20 or Donchian low broken
            if close[i] < ema20[i] or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above EMA20 or Donchian high broken
            if close[i] > ema20[i] or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Squeeze_Breakout_Pullback_v1"
timeframe = "4h"
leverage = 1.0