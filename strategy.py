#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Price breaks through Donchian channels with volume confirmation
continue in the direction of the dominant weekly trend. Weekly pivot levels
provide institutional reference points that act as support/resistance.
In bull markets: buy breakouts above weekly pivot resistance.
In bear markets: sell breakdowns below weekly pivot support.
This strategy targets 15-25 trades/year to minimize fee decay while capturing
strong momentum moves aligned with weekly structure.
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Using last completed weekly bar
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get 12h data for trend filter (optional confirmation)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        # Fallback to price action if not enough 12h data
        ema12_12h = np.full(n, np.nan)
    else:
        ema12_12h = pd.Series(df_12h['close'].values).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema12_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        
        # Determine weekly bias based on price relative to pivot
        # Above pivot = bullish bias, Below pivot = bearish bias
        weekly_bias = 1 if price > pivot_aligned[i] else -1
        
        if position == 0:
            # Long conditions: breakout above Donchian high with volume
            # AND weekly bias bullish OR price above weekly R1 (strong bullish)
            if (price > highest_high[i] and vol_ok and 
                (weekly_bias == 1 or price > r1_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below Donchian low with volume
            # AND weekly bias bearish OR price below weekly S1 (strong bearish)
            elif (price < lowest_low[i] and vol_ok and 
                  (weekly_bias == -1 or price < s1_aligned[i])):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price returns to Donchian mid-point or weekly S1
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if price < donchian_mid or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price returns to Donchian mid-point or weekly R1
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if price > donchian_mid or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0