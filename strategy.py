#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels for trend direction and 6h ATR-based volatility breakout for entry.
# Weekly pivot levels (calculated from prior week's high/low/close) provide robust support/resistance levels.
# ATR-based breakout (price > close + 1.5*ATR or < close - 1.5*ATR) captures volatility expansion.
# Only trade in direction of weekly pivot trend: long if price above weekly pivot, short if below.
# Volume confirmation (>1.3x 20-period average) reduces false breakouts.
# Designed to work in both bull and bear markets by using weekly pivot as trend filter.
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (High + Low + Close) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    support_1w = (2 * pivot_1w) - high_1w
    resistance_1w = (2 * pivot_1w) - low_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    support_aligned = align_htf_to_ltf(prices, df_1w, support_1w)
    resistance_aligned = align_htf_to_ltf(prices, df_1w, resistance_1w)
    
    # Load daily data for ATR calculation (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need volume MA and ATR
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price relative to weekly pivot
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        # Volatility breakout: price moves beyond ATR threshold from close
        upper_break = close[i] > close[i-1] + 1.5 * atr_aligned[i]
        lower_break = close[i] < close[i-1] - 1.5 * atr_aligned[i]
        
        if position == 0:
            # Long: price above weekly pivot AND upward volatility breakout
            if (above_pivot and 
                upper_break and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below weekly pivot AND downward volatility breakout
            elif (below_pivot and 
                  lower_break and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or downward breakout
            if (close[i] <= pivot_aligned[i] or 
                lower_break):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot or upward breakout
            if (close[i] >= pivot_aligned[i] or 
                upper_break):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wPivot_6hATR_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0