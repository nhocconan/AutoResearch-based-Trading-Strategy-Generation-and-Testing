#!/usr/bin/env python3
"""
1h_Keltner_Squeeze_Breakout
Hypothesis: On 1h timeframe, when price breaks above Keltner upper band during low volatility (BBW < 30th percentile) with volume > 1.5x 20-period average, go long. 
Break below lower band with volume surge goes short. 
Use 4h EMA50 as trend filter: only long when price > EMA50, short when price < EMA50. 
Apply session filter (08-20 UTC) to avoid low-liquidity hours. 
Target 15-37 trades/year (60-150 total over 4 years) with tight entries to minimize fee drag.
Works in bull via momentum breaks and bear via mean-reversion at extremes with trend filter.
"""

name = "1h_Keltner_Squeeze_Breakout"
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

    # Get 4h data (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Calculate Bollinger Band width (20, 2) for squeeze filter on 1h
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values

    # Calculate Keltner Channel (20, 2) on 1h
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    sma20_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    upper_keltner = sma20_ma + 2 * atr
    lower_keltner = sma20_ma - 2 * atr

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Get aligned values for current 1h bar
        ema50 = ema50_4h_aligned[i]
        bb_rank = bb_width_rank[i]
        upper_kelt = upper_keltner[i]
        lower_kelt = lower_keltner[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(bb_rank) or 
            np.isnan(upper_kelt) or np.isnan(lower_kelt) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 30% (contraction)
        if bb_rank > 0.3:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Keltner upper + price > EMA50 + volume surge
            if (close[i] > upper_kelt and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Keltner lower + price < EMA50 + volume surge
            elif (close[i] < lower_kelt and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Keltner lower or price < EMA50
            if (close[i] < lower_kelt or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Keltner upper or price > EMA50
            if (close[i] > upper_kelt or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals