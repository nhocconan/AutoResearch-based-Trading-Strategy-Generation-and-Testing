#161123
#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dEMA34_Volume
# Hypothesis: 4h Donchian(20) breakout in direction of 1d EMA34 trend, confirmed by volume spike (>1.5x SMA20).
# Works in bull/bear by following higher timeframe trend. Target: 20-50 trades/year to minimize fee drag.
# Uses 4h ATR for volatility filter and volatility-based position sizing.

name = "4h_Donchian20_Breakout_1dEMA34_Volume"
timeframe = "4h"
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

    # Get 4h data for price action and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 4h ATR(10) for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)

    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = align_htf_to_ltf(prices, df_4h, high_max)
    donchian_lower = align_htf_to_ltf(prices, df_4h, low_min)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i]) or
            np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian upper in 1d uptrend with volume spike and volatility filter
            if (close[i] > donchian_upper[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_sma20[i] * 1.5 and
                atr_aligned[i] > 0):  # Avoid zero ATR
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian lower in 1d downtrend with volume spike and volatility filter
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_sma20[i] * 1.5 and
                  atr_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (reversal signal) or trend change
            if close[i] < donchian_lower[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (reversal signal) or trend change
            if close[i] > donchian_upper[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals