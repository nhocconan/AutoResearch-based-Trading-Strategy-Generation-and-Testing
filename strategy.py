#!/usr/bin/env python3
# Experiment #6453: 4h Donchian(20) breakout + 12h EMA200 trend filter + volume confirmation
# Hypothesis: Donchian breakouts capture strong momentum, 12h EMA200 filters counter-trend noise, volume confirms conviction.
# Works in bull (breakouts with volume) and bear (short breakdowns with volume). Target: 20-40 trades/year.

name = "exp_6453_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # === Indicators on primary timeframe (4h) ===
    # Donchian channels (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # ATR for volatility and stoploss (14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # === HTF: 12h EMA200 trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)

    # === Signal generation ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0

    for i in range(200, n):  # warmup for 200-period EMA
        # Skip if volume is low (< 50% of average)
        if volume[i] < 0.5 * vol_ma[i]:
            continue

        # Determine HTF trend bias
        trend_bias = 0
        if close[i] > ema_200_12h_aligned[i]:
            trend_bias = 1   # bullish bias
        elif close[i] < ema_200_12h_aligned[i]:
            trend_bias = -1  # bearish bias

        # Long condition: price breaks above Donchian upper + bullish bias + volume spike
        if position == 0 and \
           close[i] > highest_high[i] and \
           trend_bias == 1 and \
           volume[i] > 1.5 * vol_ma[i]:
            signals[i] = 0.30  # long 30%
            position = 1
            entry_price = close[i]

        # Short condition: price breaks below Donchian lower + bearish bias + volume spike
        elif position == 0 and \
             close[i] < lowest_low[i] and \
             trend_bias == -1 and \
             volume[i] > 1.5 * vol_ma[i]:
            signals[i] = -0.30  # short 30%
            position = -1
            entry_price = close[i]

        # Exit conditions for long position
        elif position == 1:
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: Donchian lower break or opposite signal
            elif close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0

        # Exit conditions for short position
        elif position == -1:
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: Donchian upper break or opposite signal
            elif close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0

    return signals