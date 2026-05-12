#!/usr/bin/env python3
# 6h_Aroon_Breakout_1dTrend_Volume
# Hypothesis: Aroon (25) detects trend strength and breakout potential; long when Aroon-Up crosses above Aroon-Down with volume spike and daily uptrend, short when Aroon-Down crosses above Aroon-Up with volume spike and daily downtrend.
# Uses 1-day EMA50 for trend filter and volume > 2x 20-period average for confirmation. Designed for 6h timeframe to balance signal frequency and noise.
# Works in bull markets via momentum continuation and in bear markets via mean-reversion bounces at key levels.

name = "6h_Aroon_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Aroon (25-period)
    # Aroon-Up = ((period - periods since highest high) / period) * 100
    # Aroon-Down = ((period - periods since lowest low) / period) * 100
    period = 25
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).apply(lambda x: x.argmax(), raw=False)
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).apply(lambda x: x.argmin(), raw=False)
    aroon_up = ((period - highest_high) / period) * 100
    aroon_down = ((period - lowest_low) / period) * 100
    aroon_up = aroon_up.values
    aroon_down = aroon_down.values

    # Volume confirmation: current volume > 2x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after Aroon warmup
        # Skip if any required data is NaN
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Aroon-Up crosses above Aroon-Down with volume spike and daily uptrend
            if i > 0 and not np.isnan(aroon_up[i-1]) and not np.isnan(aroon_down[i-1]):
                if (aroon_up[i] > aroon_down[i] and aroon_up[i-1] <= aroon_down[i-1] and 
                    volume_ok[i] and price_above_daily_ema):
                    signals[i] = 0.25
                    position = 1
            # SHORT: Aroon-Down crosses above Aroon-Up with volume spike and daily downtrend
            elif i > 0 and not np.isnan(aroon_up[i-1]) and not np.isnan(aroon_down[i-1]):
                if (aroon_down[i] > aroon_up[i] and aroon_down[i-1] <= aroon_up[i-1] and 
                    volume_ok[i] and price_below_daily_ema):
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Aroon-Down crosses above Aroon-Up or trend turns down
            if i > 0 and not np.isnan(aroon_up[i-1]) and not np.isnan(aroon_down[i-1]):
                if (aroon_down[i] > aroon_up[i] and aroon_down[i-1] <= aroon_up[i-1]) or not price_above_daily_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # EXIT SHORT: Aroon-Up crosses above Aroon-Down or trend turns up
            if i > 0 and not np.isnan(aroon_up[i-1]) and not np.isnan(aroon_down[i-1]):
                if (aroon_up[i] > aroon_down[i] and aroon_up[i-1] <= aroon_down[i-1]) or not price_below_daily_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0

    return signals