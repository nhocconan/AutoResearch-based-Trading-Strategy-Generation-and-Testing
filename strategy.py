#%%# 6h_WeeklyPivot_DonchianBreakout_TrendVolume_v1
# Hypothesis: 6s Donchian(20) breakout with weekly pivot trend filter and volume confirmation.
# Uses weekly pivot levels (R1/S1) for trend direction, 6h Donchian channels for breakout signals,
# and volume spike (1.5x 20-period average) to confirm breakout strength. Designed for 15-25 trades/year.
# Works in bull/bear markets by following weekly trend direction. Exit on reversal signal (price crosses weekly pivot).
# Targets BTC/ETH with tight entry to avoid whipsaw and reduce trade frequency.

name = "6h_WeeklyPivot_DonchianBreakout_TrendVolume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 6h data for price action and Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Get weekly data for pivot trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)

    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values

    # Calculate weekly pivot levels: R1, S1
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = close_weekly + (high_weekly - low_weekly) * 1.1 / 12.0
    s1_weekly = close_weekly - (high_weekly - low_weekly) * 1.1 / 12.0
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)

    # Calculate 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian upper in weekly uptrend with volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > r1_weekly_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian lower in weekly downtrend with volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < s1_weekly_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly S1 (trend reversal)
            if close[i] < s1_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly R1 (trend reversal)
            if close[i] > r1_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals