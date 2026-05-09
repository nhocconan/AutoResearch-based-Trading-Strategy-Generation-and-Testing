#!/usr/bin/env python3
# 4h_Weekly_High_Low_Squeeze
# Hypothesis: Uses weekly high/low from prior week as breakout levels, combined with 4h Bollinger Band squeeze (low volatility) and volume confirmation.
# Works in bull markets by buying breakouts above weekly high, and in bear markets by selling breakdowns below weekly low.
# The squeeze filter ensures entries occur after low volatility, reducing false breakouts. Target: 20-35 trades/year.

name = "4h_Weekly_High_Low_Squeeze"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for high/low levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    
    # Previous week's high and low
    prev_week_high = np.roll(high_w, 1)
    prev_week_low = np.roll(low_w, 1)
    prev_week_high[0] = np.nan  # First week has no prior
    prev_week_low[0] = np.nan
    
    # Align weekly levels to 4h
    prev_week_high_aligned = align_htf_to_ltf(prices, df_w, prev_week_high)
    prev_week_low_aligned = align_htf_to_ltf(prices, df_w, prev_week_low)
    
    # 4h Bollinger Band width for squeeze detection (20, 2)
    sma_20 = np.full_like(close, np.nan)
    std_20 = np.full_like(close, np.nan)
    
    if len(close) >= 20:
        sma_20[19] = np.mean(close[0:20])
        std_20[19] = np.std(close[0:20])
        for i in range(20, len(close)):
            sma_20[i] = (sma_20[i-1] * 19 + close[i]) / 20
            std_20[i] = np.sqrt((std_20[i-1]**2 * 19 + (close[i] - sma_20[i])**2) / 20)
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    if len(bb_width) >= 40:
        for i in range(39, len(bb_width)):
            window = bb_width[i-19:i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) >= 10:
                current_val = bb_width[i]
                if not np.isnan(current_val):
                    percentile = np.sum(valid_window <= current_val) / len(valid_window) * 100
                    bb_width_percentile[i] = percentile
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(prev_week_high_aligned[i]) or np.isnan(prev_week_low_aligned[i]) or \
           np.isnan(bb_width_percentile[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: low volatility (BB width in lowest 30% percentile)
        squeeze_condition = bb_width_percentile[i] < 30
        
        if position == 0:
            # Enter long: Price breaks above prior week's high AND volume confirmation AND volatility squeeze
            if close[i] > prev_week_high_aligned[i] and volume_ratio[i] > 2.0 and squeeze_condition:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below prior week's low AND volume confirmation AND volatility squeeze
            elif close[i] < prev_week_low_aligned[i] and volume_ratio[i] > 2.0 and squeeze_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below prior week's low OR volatility expansion (end of squeeze)
            if close[i] < prev_week_low_aligned[i] or bb_width_percentile[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above prior week's high OR volatility expansion (end of squeeze)
            if close[i] > prev_week_high_aligned[i] or bb_width_percentile[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals