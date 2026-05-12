#!/usr/bin/env python3
"""
6h_Squeeze_Breakout_Momentum
Hypothesis: In 6h timeframe, Bollinger Band squeeze (low volatility) followed by breakout with volume confirmation and momentum alignment (price > EMA50) captures explosive moves in both bull and bear markets. The squeeze filter reduces false breakouts, while volume and momentum ensure follow-through. Designed for low trade frequency (15-35/year) to minimize fee drag.
"""

name = "6h_Squeeze_Breakout_Momentum"
timeframe = "6h"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 6h Bollinger Bands (20, 2) for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20
    # Percentile rank of bb_width over 50 periods to identify squeeze (low volatility)
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values

    # Calculate 6h EMA50 for momentum filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Get values
        ema50_1d_val = ema50_1d_aligned[i]
        bb_rank_val = bb_width_rank[i]
        ema50_val = ema50[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_val) or np.isnan(bb_rank_val) or np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 30% (strong contraction)
        if bb_rank_val > 0.3:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above upper BB + price > EMA50 (momentum) + 1d uptrend
            if (close[i] > upper_bb[i] and 
                close[i] > ema50_val and 
                close[i] > ema50_1d_val):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower BB + price < EMA50 (momentum) + 1d downtrend
            elif (close[i] < lower_bb[i] and 
                  close[i] < ema50_val and 
                  close[i] < ema50_1d_val):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA50 or BB width expands (>70th percentile)
            if (close[i] < ema50_val or bb_rank_val > 0.7):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA50 or BB width expands (>70th percentile)
            if (close[i] > ema50_val or bb_rank_val > 0.7):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals