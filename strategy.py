# 1h_4h1d_Camarilla_R1S1_Breakout_TrendVolume
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Uses higher timeframe (4h/1d) for signal direction and trend filtering to avoid counter-trend trades.
# 1h timeframe used only for precise entry timing on breakouts.
# Volume spike confirms breakout strength. Designed for 15-37 trades/year to minimize fee drag.
# Works in bull/bear by following 4h trend direction.

name = "1h_4h1d_Camarilla_R1S1_Breakout_TrendVolume"
timeframe = "1h"
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

    # Get 4h data for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    volume_1d = df_1d['volume'].values

    # Calculate previous 4h Camarilla levels (using prior completed 4h bar)
    # For first bar, use available data
    if len(high_4h) >= 2:
        ph = high_4h[-2]  # previous 4h high
        pl = low_4h[-2]   # previous 4h low
        pc = close_4h[-2] # previous 4h close
    else:
        ph = high_4h[0]
        pl = low_4h[0]
        pc = close_4h[0]

    range_4h = ph - pl
    camarilla_mult = 1.1 / 12
    r1 = pc + range_4h * camarilla_mult * 1
    s1 = pc - range_4h * camarilla_mult * 1

    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate 1d volume SMA20 for volume confirmation
    volume_sma20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20_1d * 1.5  # Require 1.5x average volume

    # Align all indicators to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, np.full_like(close_4h, r1))
    s1_aligned = align_htf_to_ltf(prices, df_4h, np.full_like(close_4h, s1))
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_threshold)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 in 4h uptrend with volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 in 4h downtrend with volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals