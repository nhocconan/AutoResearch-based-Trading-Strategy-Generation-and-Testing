#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Price breaking out of daily Camarilla R1/S1 levels with weekly trend confirmation and volume spike captures momentum in both bull and bear markets. Uses 12h timeframe to reduce trade frequency and avoid fee drag, with daily trend filter for alignment and volume confirmation for confirmation. Designed for 10-30 trades/year on 12h to stay within optimal range.

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

    # Calculate daily Camarilla levels (using prior day's OHLC)
    # We'll compute these on 1d data and align to 12h
    # For now, calculate directly on 12h bars using prior bar's values as proxy
    # In reality, Camarilla uses prior day's OHLC, but for 12h we use prior 12h bar
    # This is acceptable as approximation for intraday levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla R1 = Close + (High - Low) * 1.1/12
    # Camarilla S1 = Close - (High - Low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12

    # Daily trend filter: EMA34 on 1d close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        ema34_1d = np.full(len(df_1d), np.nan)
    else:
        ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Weekly trend filter: EMA34 on 1w close (for additional confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        ema34_1w = np.full(len(df_1w), np.nan)
    else:
        ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + daily uptrend + weekly uptrend + volume spike
            if (close[i] > r1[i] and 
                close[i] > ema34_1d_aligned[i] and
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + daily downtrend + weekly downtrend + volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to S1 (or close below EMA34_1d)
            if close[i] < s1[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to R1 (or close above EMA34_1d)
            if close[i] > r1[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals