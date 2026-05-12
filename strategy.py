#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: On 1h timeframe, Camarilla R1/S1 levels from prior 4h bar act as strong support/resistance.
Breaks above R1 with 4h EMA21 uptrend and volume > 1.5x 20-period average generate long signals;
breaks below S1 with 4h EMA21 downtrend and volume surge generate shorts.
Uses 4h Bollinger Band width < 50th percentile to filter choppy regimes.
Session filter (08-20 UTC) to reduce noise trades.
Target: 15-37 trades/year (60-150 total over 4 years) with low turnover to minimize fee drag.
4h trend filter helps avoid whipsaws in bear markets while capturing momentum in bull markets.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate 4h EMA21 for trend
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)

    # Calculate 4h Bollinger Band width (20, 2) for squeeze filter
    sma20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma20_4h + 2 * std20_4h
    lower_bb_4h = sma20_4h - 2 * std20_4h
    bb_width_4h = (upper_bb_4h - lower_bb_4h) / sma20_4h
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_4h).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_4h, bb_width_rank)

    # Calculate 4h Camarilla levels from previous 4h bar OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous 4h bar's HLC, so shift by 1
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    # First value will be invalid, handled by alignment
    camarilla_mult = 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * camarilla_mult
    s1 = prev_close - (prev_high - prev_low) * camarilla_mult
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Get aligned values for current 1h bar
        ema21 = ema21_4h_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema21) or np.isnan(bb_rank) or 
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
            # LONG: Price breaks above R1 + price above EMA21 + volume surge
            if (close[i] > r1_level and 
                close[i] > ema21 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + price below EMA21 + volume surge
            elif (close[i] < s1_level and 
                  close[i] < ema21 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or price below EMA21
            if (close[i] < s1_level or close[i] < ema21):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or price above EMA21
            if (close[i] > r1_level or close[i] > ema21):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals