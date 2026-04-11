#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_elder_ray_trend_follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return signals
    
    # Calculate Elder Ray components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for power calculations
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Smooth the power values with EMA13
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly trend filter: price above/below weekly EMA26
    close_1w = df_1w['close'].values
    ema26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    ema26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema26_1w)
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema26_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        
        # Elder Ray conditions
        strong_bull = bull_power_aligned[i] > 0  # Buyers in control
        strong_bear = bear_power_aligned[i] < 0  # Sellers in control
        
        # Weekly trend filter
        above_weekly_ema = price_close > ema26_1w_aligned[i]
        below_weekly_ema = price_close < ema26_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Strong bull power + above weekly EMA
        if strong_bull and above_weekly_ema:
            enter_long = True
        
        # Short: Strong bear power + below weekly EMA
        if strong_bear and below_weekly_ema:
            enter_short = True
        
        # Exit conditions: opposite power signal
        exit_long = bear_power_aligned[i] > 0  # Bear power positive
        exit_short = bull_power_aligned[i] < 0  # Bull power negative
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Elder Ray trend following with weekly trend filter.
# Uses Elder Ray (Bull Power/Bear Power) from daily data to measure buying/selling pressure.
# Weekly EMA26 filter ensures trades align with higher timeframe trend.
# Enters long when Bull Power > 0 and price above weekly EMA26.
# Enters short when Bear Power < 0 and price below weekly EMA26.
# Exits when power signals reverse.
# Works in both bull and bear markets by following the weekly trend.
# Position size 0.25 manages risk. Target: 20-40 trades per year (80-160 total over 4 years).
# Elder Ray is effective in trending markets and avoids whipsaws in ranging conditions.