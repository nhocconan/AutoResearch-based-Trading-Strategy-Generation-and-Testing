#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) on 1-day with 1-week trend filter and volume confirmation.
# Breaks above R1 in uptrend or below S1 in downtrend capture institutional breakouts.
# Trend filter avoids counter-trend trades. Volume ensures institutional participation.
# Works in bull/bear by following weekly trend. Target: 15-25 trades/year.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Camarilla levels on 1D: using previous day's OHLC
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # We need previous day's data, so shift by 1
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = phigh[1] if n > 1 else high[0]
    plow[0] = plow[1] if n > 1 else low[0]
    pclose[0] = pclose[1] if n > 1 else close[0]
    
    camarilla_range = phigh - plow
    r1 = pclose + 1.1 * camarilla_range / 12
    s1 = pclose - 1.1 * camarilla_range / 12

    # Volume confirmation: current > 1.3x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1[i]) or 
            np.isnan(s1[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1w EMA50 uptrend + volume confirmation
            if (close[i] > r1[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1w EMA50 downtrend + volume confirmation
            elif (close[i] < s1[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below previous day's close (mean reversion)
            if close[i] < pclose[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above previous day's close (mean reversion)
            if close[i] > pclose[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals