#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian breakout (20-period) with volume confirmation filters for trending moves.
# Works in bull markets via long entries on weekly highs and bear markets via short entries on weekly lows.
# Volume spike ensures institutional participation, reducing false breakouts. Target: 10-25 trades/year.

name = "1d_WeeklyDonchian_Breakout_Volume"
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
    if len(df_weekly) < 20:
        return np.zeros(n)

    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values

    # Donchian upper: max(high, 20)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Donchian lower: min(low, 20)
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values

    # Align to daily timeframe (properly delayed for weekly bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)

    # Weekly trend filter: price above/below weekly EMA20
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema20)

    # Volume spike: current > 1.5x average of last 10 days
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Donchian warmup
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Donchian high + volume spike + above weekly EMA20
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > weekly_ema20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + volume spike + below weekly EMA20
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < weekly_ema20_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low or trend weakens
            if close[i] < donchian_low_aligned[i] or close[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high or trend weakens
            if close[i] > donchian_high_aligned[i] or close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals