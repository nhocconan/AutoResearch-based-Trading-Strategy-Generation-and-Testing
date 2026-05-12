#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as intraday support/resistance. 
Breakout above R1 with 12h EMA50 uptrend and volume confirmation signals bullish momentum. 
Breakdown below S1 with 12h EMA50 downtrend and volume confirmation signals bearish momentum.
Works in bull markets via breakouts and bear markets via breakdowns, with trend filter reducing false signals.
Targets 25-35 trades/year by requiring confluence of price level, trend, and volume.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla calculation (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values

    # Get 12h data for trend filter (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate Camarilla levels for each day using prior day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(len(close_daily), np.nan)
    camarilla_s1 = np.full(len(close_daily), np.nan)
    for i in range(1, len(close_daily)):
        h = high_daily[i-1]
        l = low_daily[i-1]
        c = close_daily[i-1]
        camarilla_r1[i] = c + (h - l) * 1.1 / 12
        camarilla_s1[i] = c - (h - l) * 1.1 / 12

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1) or np.isnan(s1) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + 12h uptrend + volume confirmation
            if close[i] > r1 and close[i-1] <= r1 and close[i] > ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + 12h downtrend + volume confirmation
            elif close[i] < s1 and close[i-1] >= s1 and close[i] < ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S1 or 12h trend turns down
            if close[i] < s1 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R1 or 12h trend turns up
            if close[i] > r1 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals