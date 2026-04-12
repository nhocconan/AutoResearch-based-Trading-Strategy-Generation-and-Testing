#!/usr/bin/env python3
"""
6h_1d_Elder_Ray_Reversal_v1
Hypothesis: 6h timeframe with Elder Ray (Bull/Bear Power) reversals confirmed by 1d trend.
In bull markets (price > 1d EMA50), look for Bear Power turning negative while price declining (short setup).
In bear markets (price < 1d EMA50), look for Bull Power turning positive while price rising (long setup).
This captures mean-reversion within the dominant trend, reducing whipsaw. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Elder_Ray_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Detect turning points: Bull Power turning up, Bear Power turning down
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    bull_power_up = (bull_power > bull_power_prev) & (bull_power_prev < 0)  # Turning up from negative
    bear_power_down = (bear_power < bear_power_prev) & (bear_power_prev > 0)  # Turning down from positive
    
    # Price momentum: rising/falling close
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    price_rising = close > close_prev
    price_falling = close < close_prev
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: bull/bear market based on 1d EMA50
        bull_market = close[i] > ema_1d_aligned[i]
        bear_market = close[i] < ema_1d_aligned[i]
        
        # Entry conditions: Elder Ray reversal within trend context
        long_entry = bear_market and bull_power_up[i] and price_rising[i]
        short_entry = bull_market and bear_power_down[i] and price_falling[i]
        
        # Exit conditions: opposite Elder Ray signal or trend failure
        long_exit = bear_power_down[i] or (close[i] < ema_1d_aligned[i])  # Bear power turns down or trend fails
        short_exit = bull_power_up[i] or (close[i] > ema_1d_aligned[i])   # Bull power turns up or trend fails
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals