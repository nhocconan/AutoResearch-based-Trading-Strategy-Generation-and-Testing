#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Use daily trend filter (price above/below 1d EMA34) and Camarilla R3/S3 breakout from 12h for entry.
In bull markets, price above EMA34 and breaks above R3 triggers longs; in bear markets, price below EMA34 and breaks below S3 triggers shorts.
Volume confirmation reduces false breakouts. Targets 15-25 trades/year by requiring trend alignment, level break, and volume spike.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for trend filter (EMA34) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate EMA34 on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Get 12h data for Camarilla calculation (using previous day's OHLC) ONCE before loop
    # Note: Camarilla levels are based on previous day's range, so we use shifted 1d data
    df_1d_prev = df_1d.copy()
    df_1d_prev['high'] = df_1d_prev['high'].shift(1)  # Previous day's high
    df_1d_prev['low'] = df_1d_prev['low'].shift(1)    # Previous day's low
    df_1d_prev['close'] = df_1d_prev['close'].shift(1) # Previous day's close
    
    # Calculate Camarilla levels for each 12h bar using previous day's data
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    hl_range = df_1d_prev['high'] - df_1d_prev['low']
    r3 = df_1d_prev['close'] + hl_range * 1.1 / 4
    s3 = df_1d_prev['close'] - hl_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (already aligned since based on daily)
    r3_aligned = align_htf_to_ltf(prices, df_1d_prev, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d_prev, s3.values)

    # Volume confirmation: current volume > 1.5x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA warmup
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend from 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: price above EMA34 + breaks above R3 + volume
            if price_above_ema and close[i] > r3_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below EMA34 + breaks below S3 + volume
            elif price_below_ema and close[i] < s3_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below EMA34 OR breaks below S3 (reversal)
            if price_below_ema or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above EMA34 OR breaks above R3 (reversal)
            if price_above_ema or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals