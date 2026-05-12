#!/usr/bin/env python3
# 6h_MarketStructure_WeeklyTrend_DailyVolume
# Hypothesis: Market structure (HH/HL or LH/LL) combined with weekly trend and daily volume surge identifies
# high-probability momentum moves. Long when bullish structure + weekly uptrend + volume spike.
# Short when bearish structure + weekly downtrend + volume spike. Works in both bull/bear by following
# weekly trend direction with structure confirmation. Targets 15-25 trades/year via strict conditions.

name = "6h_MarketStructure_WeeklyTrend_DailyVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Daily volume 20-period average
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)

    # Market structure: 3-bar swing high/low
    # Bullish structure: HH and HL
    # Bearish structure: LH and LL
    hh = (high > np.roll(high, 1)) & (high > np.roll(high, 2))
    ll = (low < np.roll(low, 1)) & (low < np.roll(low, 2))
    # Need confirmation: wait for next bar to confirm swing
    hh_confirmed = np.roll(hh, 1)  # confirmed on next bar
    ll_confirmed = np.roll(ll, 1)
    hh_confirmed[0] = False
    ll_confirmed[0] = False

    # Track structure state
    bullish_structure = np.zeros(n, dtype=bool)
    bearish_structure = np.zeros(n, dtype=bool)
    bullish_structure[0] = False
    bearish_structure[0] = False

    for i in range(1, n):
        if hh_confirmed[i]:
            bullish_structure[i] = True
            bearish_structure[i] = False
        elif ll_confirmed[i]:
            bullish_structure[i] = False
            bearish_structure[i] = True
        else:
            bullish_structure[i] = bullish_structure[i-1]
            bearish_structure[i] = bearish_structure[i-1]

    # Volume spike: current 6h volume > 2.0x daily average volume (scaled)
    # Approximate: 6h volume vs daily volume - daily has 4x 6h bars
    volume_spike = volume > (2.0 * vol_ma_1d_aligned / 4.0)  # adjust for timeframe difference

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA20
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]

        if position == 0:
            # LONG: Bullish structure + weekly uptrend + volume spike
            if bullish_structure[i] and weekly_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish structure + weekly downtrend + volume spike
            elif bearish_structure[i] and weekly_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Structure turns bearish or weekly trend breaks
            if not bullish_structure[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Structure turns bullish or weekly trend breaks
            if not bearish_structure[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals