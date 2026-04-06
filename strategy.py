#!/usr/bin/env python3
"""
6h Donchian breakout with weekly pivot direction and volume confirmation
Hypothesis: In BTC/ETH/SOL, weekly pivot levels define the dominant trend.
Breaking above weekly R1 with volume confirms bullish momentum; breaking below S1 confirms bearish.
Weekly pivot acts as a trend filter, reducing false breakouts in sideways markets.
Works in bull (buy breakouts above weekly R1) and bear (sell breakdowns below S1).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Weekly pivot = (H+L+C)/3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Weekly R1 = 2*P - L
    r1_w = 2 * pivot_w - low_w
    # Weekly S1 = 2*P - H
    s1_w = 2 * pivot_w - high_w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_6h = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_weekly, s1_w)
    
    # Load daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period high/low)
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    
    # Calculate rolling max/min for Donchian
    high_max = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donch_high_6h = align_htf_to_ltf(prices, df_daily, high_max)
    donch_low_6h = align_htf_to_ltf(prices, df_daily, low_min)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 100  # For weekly pivot, Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_w_6h[i]) or np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]) or
            np.isnan(donch_high_6h[i]) or np.isnan(donch_low_6h[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or breakdown below weekly S1 or Donchian low
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] <= s1_w_6h[i] or
                close[i] <= donch_low_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or breakout above weekly R1 or Donchian high
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] >= r1_w_6h[i] or
                close[i] >= donch_high_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume confirmation and weekly pivot filter
            # Long: price above weekly R1 AND Donchian high breakout with volume
            breakout_long = (close[i] > donch_high_6h[i] and
                           close[i] > r1_w_6h[i] and
                           volume[i] > vol_ema[i] * 1.5)
            # Short: price below weekly S1 AND Donchian low breakout with volume
            breakout_short = (close[i] < donch_low_6h[i] and
                            close[i] < s1_w_6h[i] and
                            volume[i] > vol_ema[i] * 1.5)
            
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals