# 4h_DonchianBreakout_VolumeTrend_v3
# Hypothesis: Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter on 4h timeframe.
# Breakouts above upper band with volume spike and price above 1d EMA50 = long
# Breakouts below lower band with volume spike and price below 1d EMA50 = short
# Exit when price crosses back through the middle of the Donchian channel (mean reversion within channel)
# Designed for 20-40 trades/year to minimize fee drift. Works in both bull and bear by capturing breakouts with trend alignment.

name = "4h_DonchianBreakout_VolumeTrend_v3"
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
    volume = prices['volume'].values

    # Donchian Channel (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(close[i-20:i])
        lower[i] = np.min(close[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    # Get 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above upper Donchian band with volume spike and price above 1d EMA50 (uptrend)
            if close[i] > upper[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower Donchian band with volume spike and price below 1d EMA50 (downtrend)
            elif close[i] < lower[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below middle of Donchian channel (mean reversion)
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above middle of Donchian channel
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals