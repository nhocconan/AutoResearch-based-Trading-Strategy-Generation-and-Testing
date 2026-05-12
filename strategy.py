#!/usr/bin/env python3
"""
6h_Stochastic_Bollinger_Bands_Reversal
Hypothesis: On 6h timeframe, mean reversion works in BTC/ETH due to overextended moves during high volatility.
Enter long when price touches lower Bollinger Band (20,2) and Stochastic %K < 20 (oversold).
Enter short when price touches upper Bollinger Band (20,2) and Stochastic %K > 80 (overbought).
Use 1d ADX < 25 to filter ranging markets where mean reversion works best.
Exit when price crosses the 20-period SMA or Stochastic crosses 50.
Target: 20-40 trades/year to minimize fee drag in ranging markets.
"""

name = "6h_Stochastic_Bollinger_Bands_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Bollinger Bands (20,2) on 6h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20

    # Stochastic Oscillator (14,3,3) on 6h
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low_14) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    k_percent = np.where((highest_high_14 - lowest_low_14) != 0, k_percent, 50)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values

    # 1d ADX(14) for ranging market filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)
    adx_1d = adx  # Already 1d values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup for indicators
        # Skip if any required value is NaN
        if (np.isnan(sma_20[i]) or np.isnan(lower_bb[i]) or np.isnan(upper_bb[i]) or
            np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Only trade in ranging markets (ADX < 25)
            if adx_aligned[i] < 25:
                # LONG: Price touches lower BB and Stochastic oversold
                if close[i] <= lower_bb[i] and k_percent[i] < 20:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price touches upper BB and Stochastic overbought
                elif close[i] >= upper_bb[i] and k_percent[i] > 80:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above SMA OR Stochastic crosses above 50
            if close[i] > sma_20[i] or k_percent[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below SMA OR Stochastic crosses below 50
            if close[i] < sma_20[i] or k_percent[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals