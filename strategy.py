#!/usr/bin/env python3
# 1d_Donchian_Breakout_20_1wTrend_VolumeConfirm
# Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation.
# Works in bull/bear by following higher timeframe trend. Target: 30-100 trades over 4 years.
# Uses 1d timeframe with 1w EMA200 for trend filter and volume >1.5x 20-period average for confirmation.
# Entry: Close breaks above/below Donchian(20) + EMA200 trend alignment + volume confirmation.
# Exit: Close crosses opposite Donchian band (20-period) to avoid whipsaws.
# Position size: 0.25 (25% of capital) to limit drawdown.

name = "1d_Donchian_Breakout_20_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need enough data for Donchian(20) and weekly EMA200
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # Calculate Donchian channels (20-period) on daily data
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: >1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):  # Start after EMA200 warmup
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper + EMA200 uptrend + volume confirmation
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + EMA200 downtrend + volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal signal)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal signal)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals