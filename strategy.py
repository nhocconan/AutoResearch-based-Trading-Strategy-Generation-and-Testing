#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: On 4h timeframe, Camarilla R1/S1 levels from prior 12h period act as strong support/resistance.
Breaks above R1 with 12h EMA50 uptrend and volume > 1.8x 20-period average generate long signals;
breaks below S1 with 12h EMA50 downtrend and volume surge generate shorts.
Uses 12h Bollinger Band width < 50th percentile to filter choppy regimes.
Targets 20-50 trades/year (80-200 total over 4 years) with low turnover to minimize fee drag.
12h trend filter helps avoid whipsaws in bear markets while capturing momentum in bull markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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

    # Get 12h data (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate 12h Bollinger Band width (20, 2) for squeeze filter
    sma20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma20_12h + 2 * std20_12h
    lower_bb_12h = sma20_12h - 2 * std20_12h
    bb_width_12h = (upper_bb_12h - lower_bb_12h) / sma20_12h
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_12h).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_12h, bb_width_rank)

    # Calculate 12h Camarilla levels from previous 12h period OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous 12h period's HLC, so shift by 1
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    # First value will be invalid, handled by alignment
    camarilla_mult = 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * camarilla_mult
    s1 = prev_close - (prev_high - prev_low) * camarilla_mult
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)

    # Volume confirmation: 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 4h bar
        ema50 = ema50_12h_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(bb_rank) or 
            np.isnan(r1_level) or np.isnan(s1_level) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 50% (contraction)
        if bb_rank > 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above EMA50 + volume surge
            if (close[i] > r1_level and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below EMA50 + volume surge
            elif (close[i] < s1_level and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or price below EMA50
            if (close[i] < s1_level or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or price above EMA50
            if (close[i] > r1_level or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals