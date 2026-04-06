#!/usr/bin/env python3
"""
4h Camarilla Pivot + Volume Spike + Choppiness Regime
Hypothesis: Price reacts to Camarilla pivot levels (H3/L3) from higher timeframe.
In ranging markets (Choppiness > 61.8), we mean-revert at these levels with volume confirmation.
In trending markets (Choppiness < 38.2), we breakout in direction of trend.
Works in both bull/bear regimes via regime filter. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data for Camarilla pivot calculation (H3, L3 levels)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: 
    # H3 = close + (high - low) * 1.1 / 4
    # L3 = close - (high - low) * 1.1 / 4
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + (range_12h * 1.1 / 4)
    camarilla_l3 = close_12h - (range_12h * 1.1 / 4)
    
    # Align to 4h timeframe (shifted by 1 for completed bars only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period) for regime detection
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR (smoothed TR)
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        
        # Max/min close over period
        max_close = pd.Series(close).rolling(window=period, min_periods=period).max().values
        min_close = pd.Series(close).rolling(window=period, min_periods=period).min().values
        
        # Choppiness formula: 100 * log10(sum(atr) / (max_close - min_close)) / log10(period)
        sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        range_close = max_close - min_close
        # Avoid division by zero
        range_close = np.where(range_close == 0, 1e-10, range_close)
        chop = 100 * np.log10(sum_atr / range_close) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 20 for vol MA, 14 for chop)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filters
        is_ranging = chop[i] > 61.8   # Chop > 61.8 = ranging (mean revert)
        is_trending = chop[i] < 38.2  # Chop < 38.2 = trending (breakout)
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches opposite Camarilla level OR stoploss
            if (close[i] >= camarilla_h3_aligned[i] or
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # Simple SL using bar range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches opposite Camarilla level OR stoploss
            if (close[i] <= camarilla_l3_aligned[i] or
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            if is_ranging:
                # In ranging market: mean reversion at Camarilla levels
                long_setup = (close[i] <= camarilla_l3_aligned[i] and vol_filter[i])
                short_setup = (close[i] >= camarilla_h3_aligned[i] and vol_filter[i])
            elif is_trending:
                # In trending market: breakout in direction of trend
                # Use 4h close vs 12h close to determine trend
                if not (np.isnan(close_12h[-1]) if len(close_12h) == 0 else False):  # Simplified trend check
                    # Actual trend: compare current 12h close to previous
                    trend_up = close_12h[-1] > close_12h[-2] if len(close_12h) >= 2 else False
                    long_setup = (close[i] >= camarilla_h3_aligned[i] and vol_filter[i] and trend_up)
                    short_setup = (close[i] <= camarilla_l3_aligned[i] and vol_filter[i] and not trend_up)
                else:
                    long_setup = short_setup = False
            else:
                # Neutral chop zone: no trade
                long_setup = short_setup = False
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals