#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_alligator_trend_follow_v1
# Uses Williams Alligator (Jaw/Teeth/Lips) from 1d and 1w to define trend regime.
# Long when price > all three lines on both 1d and 1w, short when price < all three lines.
# Filters with 6h ADX > 25 to ensure trending conditions.
# Exits when price crosses below/above Teeth line on 1d.
# Designed for low trade frequency (target: 15-25 trades/year) by requiring multi-timeframe alignment.
# Works in trending markets via Alligator alignment and avoids whipsaws via ADX filter.
# Tested on BTC/ETH/ETH: should perform in both bull and bear via trend following.

name = "6h_1d_1w_alligator_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily and weekly data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 13 or len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Alligator lines for daily
    median_price_1d = (df_1d['high'] + df_1d['low']) / 2
    jaw_1d = median_price_1d.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = median_price_1d.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = median_price_1d.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Alligator lines for weekly
    median_price_1w = (df_1w['high'] + df_1w['low']) / 2
    jaw_1w = median_price_1w.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1w = median_price_1w.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1w = median_price_1w.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Calculate 6h ADX for trend filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above all Alligator lines on both timeframes + ADX > 25
        long_condition = (close[i] > jaw_1d_aligned[i] and close[i] > teeth_1d_aligned[i] and close[i] > lips_1d_aligned[i] and
                          close[i] > jaw_1w_aligned[i] and close[i] > teeth_1w_aligned[i] and close[i] > lips_1w_aligned[i] and
                          adx[i] > 25)
        
        # Short condition: price below all Alligator lines on both timeframes + ADX > 25
        short_condition = (close[i] < jaw_1d_aligned[i] and close[i] < teeth_1d_aligned[i] and close[i] < lips_1d_aligned[i] and
                           close[i] < jaw_1w_aligned[i] and close[i] < teeth_1w_aligned[i] and close[i] < lips_1w_aligned[i] and
                           adx[i] > 25)
        
        # Exit conditions: price crosses Teeth line on 1d
        exit_long = position == 1 and close[i] < teeth_1d_aligned[i]
        exit_short = position == -1 and close[i] > teeth_1d_aligned[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals