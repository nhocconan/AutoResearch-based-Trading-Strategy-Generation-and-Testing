#!/usr/bin/env python3
"""
#100961 - 4h_PriceChannel_Breakout_Structure
Hypothesis: Breakout above prior 4h swing high (resistance) or below prior 4h swing low (support) with volume confirmation and 1d trend filter. Uses swing points from completed 4h bars to avoid look-ahead. Works in trending markets (breakouts) and ranges (mean reversion to midpoint). Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for swing points
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate swing points from completed 4h bars (avoid look-ahead)
    # Swing high: high of completed 4h bar higher than previous and next completed 4h bar
    # Swing low: low of completed 4h bar lower than previous and next completed 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Initialize swing arrays with NaN
    swing_high = np.full_like(high_4h, np.nan)
    swing_low = np.full_like(low_4h, np.nan)
    
    # Find swing points (need at least 3 bars: prev, current, next)
    for i in range(1, len(high_4h) - 1):
        # Swing high: current high > previous high AND current high > next high
        if high_4h[i] > high_4h[i-1] and high_4h[i] > high_4h[i+1]:
            swing_high[i] = high_4h[i]
        # Swing low: current low < previous low AND current low < next low
        if low_4h[i] < low_4h[i-1] and low_4h[i] < low_4h[i+1]:
            swing_low[i] = low_4h[i]
    
    # Forward fill swing levels to create resistance/support bands
    # This ensures levels persist until broken
    resistance = np.full_like(high_4h, np.nan)
    support = np.full_like(low_4h, np.nan)
    
    last_swing_high = np.nan
    last_swing_low = np.nan
    for i in range(len(high_4h)):
        if not np.isnan(swing_high[i]):
            last_swing_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_swing_low = swing_low[i]
        resistance[i] = last_swing_high
        support[i] = last_swing_low
    
    # Align swing levels to 4h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_4h, resistance)
    support_aligned = align_htf_to_ltf(prices, df_4h, support)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: break above resistance with volume and trend filter
        if (close[i] > resistance_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: break below support with volume and trend filter
        elif (close[i] < support_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: return to midpoint of channel (mean reversion)
        elif position == 1:
            midpoint = (resistance_aligned[i] + support_aligned[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            midpoint = (resistance_aligned[i] + support_aligned[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        # Hold flat
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_PriceChannel_Breakout_Structure"
timeframe = "4h"
leverage = 1.0