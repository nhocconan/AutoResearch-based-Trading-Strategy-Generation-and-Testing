#!/usr/bin/env python3
# 4h_Donchian_Breakout_Pullback_1dTrend_Volume
# Hypothesis: Enter on pullbacks within Donchian(20) channels in the direction of 1d EMA50 trend,
# confirmed by volume spikes. Uses tight entry conditions to limit trades (target: 20-40/year).
# Works in bull via breakout pulls back up, in bear via breakdown pulls back down.

name = "4h_Donchian_Breakout_Pullback_1dTrend_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2

    # Volume confirmation: current volume > 2.0x average of last 12 periods (2 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]

        if position == 0:
            # LONG: Pullback to support in uptrend with volume
            # Condition: price near Donchian low (within 10% of range) AND uptrend AND volume spike
            range_size = donchian_high[i] - donchian_low[i]
            if range_size > 0:
                proximity_to_low = (close[i] - donchian_low[i]) / range_size
                if proximity_to_low < 0.1 and price_above_ema and volume_ok[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Pullback to resistance in downtrend with volume
            # Condition: price near Donchian high (within 10% of range) AND downtrend AND volume spike
                elif proximity_to_low > 0.9 and price_below_ema and volume_ok[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below midpoint OR trend down OR volume dries up
            if close[i] < donchian_mid[i] or not price_above_ema or not volume_ok[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above midpoint OR trend up OR volume dries up
            if close[i] > donchian_mid[i] or not price_below_ema or not volume_ok[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals