#!/usr/bin/env python3
# 6h_WeeklyPivot_PriceAction_Reversal
# Hypothesis: Price rejecting weekly pivot levels (R1/S1) with momentum exhaustion (RSI divergence) captures reversals in both bull and bear markets.
# Uses weekly pivots as key support/resistance and RSI divergence for confirmation.
# Entry: Long when price > weekly S1 + RSI bullish divergence; Short when price < weekly R1 + RSI bearish divergence.
# Exit: Mean reversion to weekly pivot point (PP) to avoid overstaying in extended moves.
# Target: 20-40 trades/year on 6h to stay within optimal range while capturing significant reversals.

name = "6h_WeeklyPivot_PriceAction_Reversal"
timeframe = "6h"
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

    # Get weekly data (as HTF for weekly pivots)
    df_weekly = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    # Align to 6h timeframe with 1-bar delay (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)

    # RSI for divergence detection
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Check for RSI divergence
            bullish_div = False
            bearish_div = False
            if i >= 3:
                # Bullish divergence: price makes lower low, RSI makes higher low
                if (close[i] < close[i-2] and 
                    close[i-2] < close[i-4] if i >= 4 else False and
                    rsi[i] > rsi[i-2] and 
                    rsi[i-2] > rsi[i-4] if i >= 4 else False):
                    bullish_div = True
                # Bearish divergence: price makes higher high, RSI makes lower high
                elif (close[i] > close[i-2] and 
                      close[i-2] > close[i-4] if i >= 4 else False and
                      rsi[i] < rsi[i-2] and 
                      rsi[i-2] < rsi[i-4] if i >= 4 else False):
                    bearish_div = True

            # LONG: Price above S1 + bullish RSI divergence
            if (close[i] > s1_aligned[i] and bullish_div):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below R1 + bearish RSI divergence
            elif (close[i] < r1_aligned[i] and bearish_div):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to weekly pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to weekly pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals