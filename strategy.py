#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_6hEMA20_Crossover
Hypothesis: In 6h timeframe, use weekly pivot points (from prior week) to determine institutional bias, combined with 6h EMA20 crossover for entry timing. Weekly pivot provides macro direction (long above weekly PP, short below), while EMA20 crossover captures momentum shifts. This reduces false signals by requiring alignment between weekly structure and short-term trend. Designed for low trade frequency (~15-30/year) to minimize fee drag in ranging/bear markets like 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly Pivot Points from prior week ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot (PP), Resistance 1 (R1), Support 1 (S1)
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pp - low_1w
    weekly_s1 = 2 * weekly_pp - high_1w
    
    # Align weekly levels to 6h timeframe (completed weekly bar only)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === 6h EMA20 for entry timing ===
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly pivot not ready (first week)
        if np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_now = ema_20[i]
        ema_prev = ema_20[i-1] if i > 0 else ema_now
        pp = weekly_pp_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        if position == 0:
            # Long: price above weekly PP AND EMA20 bullish crossover (EMA20 crosses above price)
            if price_close > pp and ema_now > price_close and ema_prev <= prices['close'].iloc[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly PP AND EMA20 bearish crossover (EMA20 crosses below price)
            elif price_close < pp and ema_now < price_close and ema_prev >= prices['close'].iloc[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite EMA20 crossover or price revisits weekly PP
            if position == 1:
                # Exit long: EMA20 bearish crossover or price drops below weekly PP
                if ema_now < price_close and ema_prev >= prices['close'].iloc[i-1] or price_close < pp:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: EMA20 bullish crossover or price rises above weekly PP
                if ema_now > price_close and ema_prev <= prices['close'].iloc[i-1] or price_close > pp:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Direction_6hEMA20_Crossover"
timeframe = "6h"
leverage = 1.0