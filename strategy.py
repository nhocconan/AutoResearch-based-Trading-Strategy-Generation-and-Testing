#/usr/bin/env python3
# 6h_Donchian_20_Breakout_WeeklyTrend_1dVolume
# Hypothesis: 6h Donchian(20) breakouts with weekly trend filter and daily volume confirmation.
# Works in bull/bear by filtering breakouts with weekly trend and requiring volume surge.
# Weekly trend: price above/below weekly EMA50.
# Volume: current 6h volume > 2.0x 20-period SMA.
# Entry: Long when price breaks above Donchian(20) high + weekly uptrend + volume spike.
#        Short when price breaks below Donchian(20) low + weekly downtrend + volume spike.
# Exit: Reverse signal on opposite Donchian breakout.
# Position size: 0.25.

name = "6h_Donchian_20_Breakout_WeeklyTrend_1dVolume"
timeframe = "6h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Daily volume 20-period SMA for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma20_1d)

    # Donchian(20) channels: 20-period high/low
    # Calculate using rolling window on price data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 6h bar
        ema50_aligned = ema50_1w_aligned[i]
        vol_sma20_aligned = volume_sma20_1d_aligned[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned) or np.isnan(vol_sma20_aligned) or
            np.isnan(dch_high) or np.isnan(dch_low)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume confirmation: current volume > 2.0x daily 20-period SMA
        volume_filter = volume[i] > 2.0 * vol_sma20_aligned

        if position == 0:
            # LONG: Price breaks above Donchian high + weekly uptrend + volume spike
            if (close[i] > dch_high and
                close[i] > ema50_aligned and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly downtrend + volume spike
            elif (close[i] < dch_low and
                  close[i] < ema50_aligned and
                  volume_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < dch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > dch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals