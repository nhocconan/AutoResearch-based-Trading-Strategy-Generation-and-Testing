#!/usr/bin/env python3
# 1d_WilliamsAlligator_Trend_Filter
# Hypothesis: Williams Alligator (3 SMAs: Jaw 13, Teeth 8, Lips 5) identifies trend strength on daily timeframe.
# In bull markets: price above all three lines with Lips > Teeth > Jaw (bullish alignment) → long.
# In bear markets: price below all three lines with Lips < Teeth < Jaw (bearish alignment) → short.
# Weekly trend filter: only trade in direction of weekly EMA(34) to avoid counter-trend whipsaws.
# Volume confirmation: current volume > 1.5x 20-day average ensures institutional participation.
# Designed for 10-20 trades/year to minimize fee drag. Works in both bull and bear by capturing strong trends.

name = "1d_WilliamsAlligator_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Williams Alligator SMAs (13, 8, 5 periods)
    jaw = np.zeros(n)
    teeth = np.zeros(n)
    lips = np.zeros(n)
    
    # Jaw: 13-period SMA of median price
    median_price = (high + low) / 2
    for i in range(12, n):
        jaw[i] = np.mean(median_price[i-12:i+1])
    
    # Teeth: 8-period SMA of median price
    for i in range(7, n):
        teeth[i] = np.mean(median_price[i-7:i+1])
    
    # Lips: 5-period SMA of median price
    for i in range(4, n):
        lips[i] = np.mean(median_price[i-4:i+1])

    # Weekly trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = np.zeros_like(weekly_close)
    for i in range(33, len(weekly_close)):
        if i == 33:
            weekly_ema[i] = np.mean(weekly_close[:34])
        else:
            weekly_ema[i] = (weekly_close[i] * 2 + weekly_ema[i-1] * 32) / 34
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)

    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros(n)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        bullish_alignment = (lips[i] > teeth[i] > jaw[i]) and (close[i] > lips[i])
        bearish_alignment = (lips[i] < teeth[i] < jaw[i]) and (close[i] < lips[i])
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]

        if position == 0:
            # LONG: bullish alignment + weekly uptrend + volume spike
            if bullish_alignment and weekly_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish alignment + weekly downtrend + volume spike
            elif bearish_alignment and weekly_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish alignment or weekly trend turns down
            if bearish_alignment or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish alignment or weekly trend turns up
            if bullish_alignment or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals