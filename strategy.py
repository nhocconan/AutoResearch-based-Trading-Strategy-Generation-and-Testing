#!/usr/bin/env python3
"""
1d_WeeklyPivot_PriceChannelBreakout_VolumeSpike
Hypothesis: Price breaking above/below weekly pivot resistance/support levels (calculated from prior week OHLC) with 1d EMA trend filter and volume confirmation (1.5x average) captures strong trending moves while avoiding false breakouts. Weekly pivots represent stronger support/resistance than daily, reducing false signals. Works in bull/bear by following 1d trend direction. Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
"""

name = "1d_WeeklyPivot_PriceChannelBreakout_VolumeSpike"
timeframe = "1d"
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

    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')

    # Calculate Weekly Pivot levels: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
    # Using prior week's data
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values

    # Shift by 1 to use previous week's data
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close[0] = np.nan
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan

    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_r1 = 2 * pivot - prev_weekly_low  # Resistance 1
    weekly_s1 = 2 * pivot - prev_weekly_high  # Support 1

    # Align Weekly Pivot levels to 1d timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume spike: >1.5x 20-period average (1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Weekly R1 + 1d EMA50 uptrend + volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema_50_1d[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Weekly S1 + 1d EMA50 downtrend + volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema_50_1d[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Weekly S1 (reversal level)
            if close[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Weekly R1 (reversal level)
            if close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals