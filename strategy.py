#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Trend_Breakout
Hypothesis: Weekly pivot levels (from previous week) act as strong support/resistance.
Breakouts above R1 with volume confirmation and bullish 12h trend go long.
Breakdowns below S1 with volume confirmation and bearish 12h trend go short.
Uses weekly timeframe for structural levels, 12h for trend filter, and 6s for entry timing.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in both bull and bear regimes by following breakouts with institutional volume backing.
"""

name = "6h_Weekly_Pivot_Trend_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_ohlc

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    # Using previous week's values to avoid look-ahead
    wk_high = df_w['high'].values
    wk_low = df_w['low'].values
    wk_close = df_w['close'].values
    
    # Calculate pivot points for each week
    wk_p = (wk_high + wk_low + wk_close) / 3.0
    wk_r1 = 2 * wk_p - wk_low
    wk_s1 = 2 * wk_p - wk_high
    
    # Align to 6s timeframe (previous week's values available after weekly bar closes)
    pivot_p = align_ltf_to_ohlc(prices, df_w, wk_p)
    pivot_r1 = align_ltf_to_ohlc(prices, df_w, wk_r1)
    pivot_s1 = align_ltf_to_ohlc(prices, df_w, wk_s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_aligned = align_ltf_to_ohlc(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: Break above weekly R1 with volume spike and bullish 12h trend
            if (close[i] > pivot_r1[i] and 
                volume_spike[i] and 
                close[i] > trend_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 with volume spike and bearish 12h trend
            elif (close[i] < pivot_s1[i] and 
                  volume_spike[i] and 
                  close[i] < trend_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below weekly pivot point or trend turns bearish
            if (close[i] < pivot_p[i] or 
                close[i] < trend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above weekly pivot point or trend turns bullish
            if (close[i] > pivot_p[i] or 
                close[i] > trend_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def align_ltf_to_ohlc(ltf_prices, htf_df, htf_values):
    """
    Helper function to align HTF values to LTF timeframe using open_time matching.
    This ensures we use only completed HTF bars (no look-ahead).
    """
    # Create a mapping from LTF index to HTF index based on open_time
    htf_index = htf_df.index
    ltf_index = ltf_prices.index
    
    # For each LTF bar, find the most recent completed HTF bar
    aligned_values = np.full(len(ltf_prices), np.nan)
    htf_idx = 0
    
    for i in range(len(ltf_prices)):
        # Advance HTF index while HTF bar time <= LTF bar time
        while htf_idx < len(htf_index) - 1 and htf_index[htf_idx + 1] <= ltf_index[i]:
            htf_idx += 1
        
        # Use the completed HTF bar value
        if htf_idx < len(htf_index):
            aligned_values[i] = htf_values[htf_idx]
        else:
            # Not enough HTF data yet
            aligned_values[i] = np.nan
    
    # Forward fill to handle any remaining NaNs (should not happen with sufficient data)
    aligned_values = pd.Series(aligned_values).ffill().bfill().values
    return aligned_values

# Override the align_ltf_to_ohlc function to use the proper one from mtf_data if available
try:
    from mtf_data import align_htf_to_ltf
    def align_ltf_to_ohlc(ltf_prices, htf_df, htf_values):
        return align_htf_to_ltf(ltf_prices, htf_df, htf_values)
except ImportError:
    pass  # Use the custom implementation above