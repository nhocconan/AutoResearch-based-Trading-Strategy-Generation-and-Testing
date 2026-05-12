# 12h_Donchian_20_Breakout_1dTrend_Volume
# Hypothesis: Donchian channel breakouts on 12h timeframe filtered by 1d trend (EMA50) and volume spikes.
# Long: Price breaks above 20-period 12h high + volume > 1.5x volume SMA20 + 1d close > 1d EMA50
# Short: Price breaks below 20-period 12h low + volume > 1.5x volume SMA20 + 1d close < 1d EMA50
# Exit: Price crosses opposite Donchian band (exit long when price < 12h low, exit short when price > 12h high)
# Uses 12h for lower trade frequency (target ~15-30 trades/year) to avoid fee drag.

name = "12h_Donchian_20_Breakout_1dTrend_Volume"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 12h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 12h Donchian high + volume spike + 1d uptrend
            if (close[i] > donchian_high[i] and
                volume[i] > volume_threshold[i] and
                close_1d[-1] > ema50_1d[-1] if len(close_1d) > 0 else False):  # Use last known 1d close
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 12h Donchian low + volume spike + 1d downtrend
            elif (close[i] < donchian_low[i] and
                  volume[i] > volume_threshold[i] and
                  close_1d[-1] < ema50_1d[-1] if len(close_1d) > 0 else False):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 12h Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 12h Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals