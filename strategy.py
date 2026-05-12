#!/usr/bin/env python3
"""
6h_OrderBlock_Retest_1dTrend_Filtered
Hypothesis: On 6h timeframe, enter long when price retraces to a bullish order block (last down candle before strong up move) in an uptrend (1d EMA50 rising), and short when price retraces to a bearish order block (last up candle before strong down move) in a downtrend (1d EMA50 falling). Uses volume confirmation to validate institutional interest. Designed to work in both bull (buy dips) and bear (sell rallies) markets by following the 1d trend. Targets 15-25 trades/year to minimize fee drag.
"""

name = "6h_OrderBlock_Retest_1dTrend_Filtered"
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

    # Get 1d data for trend and order blocks
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_prev)

    # Identify order blocks on 1d timeframe
    # Bullish OB: last down candle before a strong up move (close > open and next candle closes > its high)
    # Bearish OB: last up candle before a strong down move (close < open and next candle closes < its low)
    bullish_ob_top = np.full(len(close_1d), np.nan)
    bullish_ob_bottom = np.full(len(close_1d), np.nan)
    bearish_ob_top = np.full(len(close_1d), np.nan)
    bearish_ob_bottom = np.full(len(close_1d), np.nan)

    for i in range(1, len(close_1d)-1):
        # Bullish OB: current candle is down (close < open), next candle is strong up (close > high of current)
        if close_1d[i] < high_1d[i] and close_1d[i+1] > high_1d[i]:
            bullish_ob_top[i] = high_1d[i]    # OB top = high of the down candle
            bullish_ob_bottom[i] = low_1d[i]  # OB bottom = low of the down candle
        # Bearish OB: current candle is up (close > open), next candle is strong down (close < low of current)
        if close_1d[i] > low_1d[i] and close_1d[i+1] < low_1d[i]:
            bearish_ob_top[i] = high_1d[i]    # OB top = high of the up candle
            bearish_ob_bottom[i] = low_1d[i]  # OB bottom = low of the up candle

    # Align order block levels to 6h timeframe
    bullish_ob_top_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_top)
    bullish_ob_bottom_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_bottom)
    bearish_ob_top_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_top)
    bearish_ob_bottom_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_bottom)

    # Volume confirmation: volume > 1.5x 20-period average on 6h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bullish_ob_top_aligned[i]) or 
            np.isnan(bullish_ob_bottom_aligned[i]) or np.isnan(bearish_ob_top_aligned[i]) or
            np.isnan(bearish_ob_bottom_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price retraces to bullish OB + 1d uptrend + volume confirmation
            if (low[i] <= bullish_ob_top_aligned[i] and high[i] >= bullish_ob_bottom_aligned[i] and
                close[i] > ema50_1d_aligned[i] and  # price above EMA50 (uptrend)
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price retraces to bearish OB + 1d downtrend + volume confirmation
            elif (low[i] <= bearish_ob_top_aligned[i] and high[i] >= bearish_ob_bottom_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and  # price below EMA50 (downtrend)
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below OB bottom OR trend turns down
            if close[i] < bullish_ob_bottom_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above OB top OR trend turns up
            if close[i] > bearish_ob_top_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals