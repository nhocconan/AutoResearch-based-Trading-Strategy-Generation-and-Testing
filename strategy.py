#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with 1d EMA34 trend and volume confirmation
# Camarilla pivot levels derived from weekly OHLC provide institutional support/resistance.
# Breakout above R4 or below S4 with 1d EMA34 trend alignment and volume spike captures strong momentum.
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend).
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_WeeklyCamarilla_R4S4_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range
    range_1w = high_1w - low_1w
    # Camarilla levels
    r4_1w = pivot_1w + (range_1w * 1.5)
    s4_1w = pivot_1w - (range_1w * 1.5)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above weekly R4 AND 1d EMA34 uptrend AND volume spike
            if close[i] > r4_1w_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below weekly S4 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < s4_1w_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly pivot OR Bearish reversal
            if close[i] < pivot_1w[i] if not np.isnan(pivot_1w[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly pivot OR Bullish reversal
            if close[i] > pivot_1w[i] if not np.isnan(pivot_1w[i]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals