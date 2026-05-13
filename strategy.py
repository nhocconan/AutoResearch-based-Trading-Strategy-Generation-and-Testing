#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Buy when price breaks above Camarilla R1 level and sell when breaks below S1 on 12h timeframe,
# filtered by 1d EMA trend and volume confirmation. Camarilla levels provide institutional support/resistance.
# Works in bull/bear markets by trading only in direction of daily trend.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')

    # Calculate Camarilla levels from previous day (OHLC)
    # Since we're on 12h timeframe, we use previous 12h bar's OHLC for same-day levels
    # But for proper Camarilla, we need daily OHLC. We'll use 1d data to get daily OHLC,
    # then align the levels from previous day to current 12h bars.
    # Actually, Camarilla levels are calculated from previous day's OHLC and remain static for current day.
    # We'll compute them from 1d data and align to 12h timeframe.

    # Extract OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values

    # Calculate Camarilla levels for each day (based on previous day's OHLC)
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    # We shift by 1 to use previous day's values
    prev_close = np.roll(daily_close, 1)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    # First day will have invalid values (rolled from last), but we'll handle with valid checks later
    rng = prev_high - prev_low
    R1 = prev_close + rng * 1.1 / 12
    S1 = prev_close - rng * 1.1 / 12

    # Align Camarilla levels to 12h timeframe (they are constant throughout the day)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 level in uptrend with volume
            if (close[i] > R1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 level in downtrend with volume
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks back below R1 or trend turns down
            if close[i] < R1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks back above S1 or trend turns up
            if close[i] > S1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals