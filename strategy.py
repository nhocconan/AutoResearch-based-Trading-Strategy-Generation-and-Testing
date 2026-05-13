#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and ATR-based stoploss.
# Long when price breaks above upper Donchian channel AND close > 1w EMA200.
# Short when price breaks below lower Donchian channel AND close < 1w EMA200.
# Exit on opposite breakout or ATR(14) trailing stop (2.5x).
# Uses 1d primary timeframe with 1w trend filter for noise reduction, targeting 30-100 total trades over 4 years.
# Donchian channels provide clear structure, 1w EMA200 filters long-term trend to avoid counter-trend trades.
# Designed to work in both bull and bear markets via strict entry conditions and trend alignment.

name = "1d_Donchian20_1wEMA200_Breakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) for 1d: based on previous 20 bars to avoid look-ahead
    # Upper = max(high of past 20 bars), Lower = min(low of past 20 bars)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1w data for EMA200 trend filter (MTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w close
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF arrays to 1d timeframe (wait for completed 1w bar)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND close > 1w EMA200
            if close[i] > donchian_upper[i] and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price breaks below lower Donchian AND close < 1w EMA200
            elif close[i] < donchian_lower[i] and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: price breaks below lower Donchian (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] < donchian_lower[i]
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if breakout_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: price breaks above upper Donchian (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] > donchian_upper[i]
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if breakout_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals