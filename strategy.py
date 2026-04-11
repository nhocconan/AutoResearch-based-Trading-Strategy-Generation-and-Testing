#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# - Long when Bull Power > 0 and rising + Bear Power < 0 (bullish momentum)
# - Short when Bear Power > 0 and rising + Bull Power < 0 (bearish momentum)
# - Williams %R from 1d acts as regime filter: only trade when not in extreme overbought/oversold
# - Avoids false signals during strong trends where Elder Ray can whipsaw
# - Works in bull markets (catch strong uptrends) and bear markets (catch strong downtrends)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h

name = "6h_1d_elder_ray_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Williams %R regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r_1d = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    # Avoid division by zero
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Slope of Elder Ray components (1-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R regime filter: avoid extreme overbought (> -20) and oversold (< -80)
        # Only trade when Williams %R is between -80 and -20 (not in extreme zones)
        williams_filter = (williams_r_aligned[i] > -80) and (williams_r_aligned[i] < -20)
        
        # Elder Ray conditions
        bull_power_current = bull_power[i]
        bear_power_current = bear_power[i]
        bull_power_rising = bull_power_slope[i] > 0
        bear_power_rising = bear_power_slope[i] > 0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 and rising + Bear Power < 0 (bullish momentum)
        if bull_power_current > 0 and bull_power_rising and bear_power_current < 0 and williams_filter:
            enter_long = True
        
        # Short: Bear Power > 0 and rising + Bull Power < 0 (bearish momentum)
        if bear_power_current > 0 and bear_power_rising and bull_power_current < 0 and williams_filter:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray signal or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power becomes positive OR regime turns extreme
            exit_long = (bear_power_current > 0) or (not williams_filter)
        elif position == -1:
            # Exit short if Bull Power becomes positive OR regime turns extreme
            exit_short = (bull_power_current > 0) or (not williams_filter)
        
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