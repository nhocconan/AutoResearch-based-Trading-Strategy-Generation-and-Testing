#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot breakouts with daily trend filter and volume confirmation work across market regimes.
# Long when price breaks above R1 with daily uptrend and volume spike.
# Short when price breaks below S1 with daily downtrend and volume spike.
# Uses daily EMA34 for trend filter and volume > 1.5x 20-period average for confirmation.
# Designed for 4h timeframe to target 20-50 trades/year with low frequency and high edge.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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

    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla R1 = C + (H-L)*1.1/12
    r1 = pc + (ph - pl) * 1.1 / 12
    # Camarilla S1 = C - (H-L)*1.1/12
    s1 = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1 with daily uptrend and volume confirmation
            if close[i] > r1_aligned[i] and price_above_daily_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with daily downtrend and volume confirmation
            elif close[i] < s1_aligned[i] and price_below_daily_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or daily trend turns down
            if close[i] < s1_aligned[i] or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or daily trend turns up
            if close[i] > r1_aligned[i] or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals