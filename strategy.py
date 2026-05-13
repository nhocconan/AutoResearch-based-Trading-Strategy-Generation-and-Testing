#!/usr/bin/env python3
# 4h_HTF_Trend_Liquidity_Zone_Breakout
# Hypothesis: Combine 4h liquidity zone breakouts (equal highs/lows) with 1d trend filter and volume confirmation.
# Long when price breaks above a 4h equal high with 1d EMA50 uptrend and volume spike.
# Short when price breaks below a 4h equal low with 1d EMA50 downtrend and volume spike.
# Exit when price returns to the opposite liquidity zone or trend changes.
# Liquidity zones act as institutional support/resistance, reducing false breakouts.
# Designed for low-moderate trade frequency (50-150 total trades over 4 years) with high-probability entries.

name = "4h_HTF_Trend_Liquidity_Zone_Breakout"
timeframe = "4h"
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

    # Get 4h data for liquidity zone detection (equal highs/lows)
    df_4h = get_htf_data(prices, '4h')
    
    # Detect equal highs (resistance) and equal lows (support) within 0.1% tolerance
    # Equal high: current high within 0.1% of previous high
    # Equal low: current low within 0.1% of previous low
    eq_high = np.where(np.abs(df_4h['high'] - df_4h['high'].shift(1)) / df_4h['high'].shift(1) < 0.001, df_4h['high'], np.nan)
    eq_low = np.where(np.abs(df_4h['low'] - df_4h['low'].shift(1)) / df_4h['low'].shift(1) < 0.001, df_4h['low'], np.nan)
    
    # Forward fill to establish zones
    eq_high_series = pd.Series(eq_high).ffill().values
    eq_low_series = pd.Series(eq_low).ffill().values
    
    # Align liquidity zones to 4h timeframe (no additional delay needed as zones are based on closed bars)
    eq_high_4h_aligned = align_htf_to_ltf(prices, df_4h, eq_high_series)
    eq_low_4h_aligned = align_htf_to_ltf(prices, df_4h, eq_low_series)

    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(eq_high_4h_aligned[i]) or np.isnan(eq_low_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above equal high + 1d EMA50 uptrend + volume spike
            if (close[i] > eq_high_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below equal low + 1d EMA50 downtrend + volume spike
            elif (close[i] < eq_low_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to equal low or trend changes (price below EMA50)
            if (close[i] <= eq_low_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to equal high or trend changes (price above EMA50)
            if (close[i] >= eq_high_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals