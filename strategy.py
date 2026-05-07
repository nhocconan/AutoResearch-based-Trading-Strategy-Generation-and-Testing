#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Long when price touches or breaks above S1 level AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when price touches or breaks below R1 level AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Camarilla range (between S1 and R1).
# This strategy targets mean reversion at key pivot levels with trend alignment.
# The 1d EMA34 filter ensures we trade with the higher timeframe trend.
# Volume confirmation reduces false breakouts and ensures institutional participation.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "12h_Camarilla_S1R1_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from previous day
    # Calculate using previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First value will be invalid, will be handled by NaN checks
    
    # Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one previous day for Camarilla calculation
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches or breaks above S1, 1d EMA34 rising, volume filter
            long_cond = (close[i] >= S1[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price touches or breaks below R1, 1d EMA34 falling, volume filter
            short_cond = (close[i] <= R1[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Camarilla range (below S1)
            if close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Camarilla range (above R1)
            if close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals