# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot levels for directional bias and daily Donchian breakout for entry timing.
Weekly pivot levels (calculated from prior week's OHLC) provide strong support/resistance that holds across market regimes.
In bull markets, price tends to respect weekly support and bounce from S1/S2; in bear markets, weekly resistance (R1/R2) acts as ceiling.
Daily Donchian breakout (20-period) provides entry timing in the direction of the weekly bias, filtering false breaks.
Volume confirmation ensures breakouts have institutional participation.
Target: 15-30 trades/year per symbol to minimize fee drag while capturing significant moves.
Works in both bull (buy dips to weekly support) and bear (sell rallies to weekly resistance) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    # Support 2: S2 = P - (H - L)
    # Resistance 2: R2 = P + (H - L)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate pivots using shifted values (prior week)
    pp = (np.roll(high_weekly, 1) + np.roll(low_weekly, 1) + np.roll(close_weekly, 1)) / 3.0
    r1 = 2 * pp - np.roll(high_weekly, 1)
    s1 = 2 * pp - np.roll(low_weekly, 1)
    r2 = pp + (np.roll(high_weekly, 1) - np.roll(low_weekly, 1))
    s2 = pp - (np.roll(high_weekly, 1) - np.roll(low_weekly, 1))
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Load daily data for Donchian breakout
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    upper_daily = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_daily = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 6h
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper_daily)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower_daily)
    
    # Volume confirmation: 6h volume vs 20-period average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any key values are not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        
        # Volume threshold: require above-average volume for confirmation
        vol_threshold = 1.5
        
        if position == 0:
            # Long conditions:
            # 1. Price breaks above daily Donchian upper (breakout)
            # 2. Price is above weekly pivot (bullish bias)
            # 3. Volume confirms the breakout
            if (price_close > upper_aligned[i] and
                price_close > pp_aligned[i] and
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            
            # Short conditions:
            # 1. Price breaks below daily Donchian lower (breakdown)
            # 2. Price is below weekly pivot (bearish bias)
            # 3. Volume confirms the breakdown
            elif (price_close < lower_aligned[i] and
                  price_close < pp_aligned[i] and
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions:
            # Long exit: price returns to weekly pivot or breaks below daily lower
            # Short exit: price returns to weekly pivot or breaks above daily upper
            if position == 1:
                if (price_close < pp_aligned[i] or
                    price_close < lower_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                if (price_close > pp_aligned[i] or
                    price_close > upper_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyDonchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0