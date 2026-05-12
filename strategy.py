#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: On daily timeframe, Camarilla R1/S1 levels from prior week act as strong support/resistance.
Breaks above R1 with weekly EMA34 uptrend and volume > 1.8x 20-period average generate long signals;
breaks below S1 with weekly EMA34 downtrend and volume surge generate shorts.
Uses weekly Bollinger Band width < 50th percentile to filter choppy regimes.
Targets 7-25 trades/year (30-100 total over 4 years) with low turnover to minimize fee drag.
Weekly trend filter helps avoid whipsaws in bear markets while capturing momentum in bull markets.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate weekly Bollinger Band width (20, 2) for squeeze filter
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma20_1w + 2 * std20_1w
    lower_bb_1w = sma20_1w - 2 * std20_1w
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / sma20_1w
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_1w).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1w, bb_width_rank)

    # Calculate weekly Camarilla levels from previous week OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous week's HLC, so shift by 1
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    # First value will be invalid, handled by alignment
    camarilla_mult = 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * camarilla_mult
    s1 = prev_close - (prev_high - prev_low) * camarilla_mult
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)

    # Volume confirmation: 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current daily bar
        ema34 = ema34_1w_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34) or np.isnan(bb_rank) or 
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
            # LONG: Price breaks above R1 + price above EMA34 + volume surge
            if (close[i] > r1_level and 
                close[i] > ema34 and 
                volume[i] > vol_avg_val * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below EMA34 + volume surge
            elif (close[i] < s1_level and 
                  close[i] < ema34 and 
                  volume[i] > vol_avg_val * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or price below EMA34
            if (close[i] < s1_level or close[i] < ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or price above EMA34
            if (close[i] > r1_level or close[i] > ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals