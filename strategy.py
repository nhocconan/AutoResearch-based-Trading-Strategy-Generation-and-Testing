#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h trend filter (price above/below 200 EMA) and 1d volume confirmation (volume > 1.5x 20-day average) provide high-probability entries. Uses 4h for trend direction, 1d for volume regime, and 1h for precise entry timing. Designed for 15-30 trades/year to minimize fee drift while capturing momentum in both bull and bear markets.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    # Calculate EMA(200) on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)

    # Get 1d data for volume filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 20-day average volume on 1d
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Camarilla levels for 1h (using prior 1h bar's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We'll calculate these using the previous bar's OHLC to avoid look-ahead
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    for i in range(1, n):
        # Use previous bar's OHLC to calculate levels for current bar
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_ = ph - pl
        r1[i] = pc + range_ * 1.1 / 12
        s1[i] = pc - range_ * 1.1 / 12

    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup for EMA
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Get aligned values
        ema_200 = ema_200_4h_aligned[i]
        vol_avg = vol_avg_20_1d_aligned[i]
        vol_1d = volume_1d[i // 24] if i // 24 < len(volume_1d) else vol_avg  # Safe 1d volume lookup
        r1_val = r1[i]
        s1_val = s1[i]

        if np.isnan(ema_200) or np.isnan(vol_avg):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 + 4h uptrend (price > EMA200) + 1d volume surge
            if close[i] > r1_val and close[i] > ema_200 and vol_1d > vol_avg * 1.5:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 + 4h downtrend (price < EMA200) + 1d volume surge
            elif close[i] < s1_val and close[i] < ema_200 and vol_1d > vol_avg * 1.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S1 or trend change (price < EMA200)
            if close[i] < s1_val or close[i] < ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Break above R1 or trend change (price > EMA200)
            if close[i] > r1_val or close[i] > ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals