#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily EMA13 for Elder Ray
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly EMA26 for trend filter
    ema26_1w = pd.Series(df_1w['close'].values).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema26_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(26, n):  # Wait for weekly EMA26
        # Skip if any data is invalid
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(ema26_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema13_1d_aligned[i]
        bear_power = low[i] - ema13_1d_aligned[i]
        
        # Trend filter from weekly EMA26
        above_weekly_ema = close[i] > ema26_1w_aligned[i]
        below_weekly_ema = close[i] < ema26_1w_aligned[i]
        
        # Entry conditions
        enter_long = bull_power > 0 and above_weekly_ema
        enter_short = bear_power < 0 and below_weekly_ema
        
        # Exit conditions: opposite Elder Ray power
        exit_long = bear_power < 0
        exit_short = bull_power > 0
        
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

# Hypothesis: Elder Ray Power (Bull/Bear) with weekly EMA trend filter on 6h timeframe.
# Bull Power = High - Daily EMA13, Bear Power = Low - Daily EMA13.
# Long when Bull Power > 0 AND price above Weekly EMA26 (uptrend).
# Short when Bear Power < 0 AND price below Weekly EMA26 (downtrend).
# Exits when opposite power appears, preventing adverse moves.
# Works in bull markets (captures strength via Bull Power) and bear markets (captures weakness via Bear Power).
# Position size 0.25 balances risk and return. Target: 50-100 total trades over 4 years (12-25/year).