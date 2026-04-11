#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long: Bull Power > 0 AND Bear Power < 0 AND 1d Williams %R < -80 (oversold)
# - Short: Bear Power > 0 AND Bull Power < 0 AND 1d Williams %R > -20 (overbought)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear strength relative to EMA, effective in trending markets
# - Williams %R on 1d provides regime filter to avoid counter-trend trades
# - Works in both bull (buy oversold pullbacks) and bear (sell overbought bounces) markets

name = "6h_1d_elder_ray_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Pre-compute EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(williams_r_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema_13[i]
        bear_power = ema_13[i] - low[i]
        
        # Williams %R regime filter
        williams_r = williams_r_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power positive, Bear Power negative, and 1d oversold
        if bull_power > 0 and bear_power < 0 and williams_r < -80:
            enter_long = True
        
        # Short: Bear Power positive, Bull Power negative, and 1d overbought
        if bear_power > 0 and bull_power < 0 and williams_r > -20:
            enter_short = True
        
        # Exit conditions: reverse signal or power divergence
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power becomes positive (bulls losing strength)
            exit_long = bear_power > 0
        elif position == -1:
            # Exit short if Bull Power becomes positive (bears losing strength)
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