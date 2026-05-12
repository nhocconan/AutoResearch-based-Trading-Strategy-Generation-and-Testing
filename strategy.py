#!/usr/bin/env python3
# 6h_WilliamsAlligator_Trend_Confirmation_v2
# Hypothesis: Use Williams Alligator (3 SMAs) on 6h for trend direction + 1d for higher timeframe trend filter.
# Long when 6h Jaw < Teeth < Lips (bullish alignment) + 1d close > 1d EMA50.
# Short when 6h Jaw > Teeth > Lips (bearish alignment) + 1d close < 1d EMA50.
# Exit when alignment breaks or 1d trend reverses.
# Williams Alligator works in both bull/bear markets by capturing strong trends while avoiding chop.

name = "6h_WilliamsAlligator_Trend_Confirmation_v2"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 6h Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    # Alligator uses SMAs with future shift, but we only use completed bar values
    jaw_6h = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_6h = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_6h = pd.Series(close).rolling(window=5, min_periods=5).mean().values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Jaw period
        # Skip if any required data is NaN
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Williams Alligator alignment signals
        bullish_alignment = jaw_6h[i] < teeth_6h[i] < lips_6h[i]
        bearish_alignment = jaw_6h[i] > teeth_6h[i] > lips_6h[i]

        if position == 0:
            # LONG: Bullish Alligator alignment + 1d uptrend
            if bullish_alignment and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment + 1d downtrend
            elif bearish_alignment and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bullish alignment breaks OR 1d trend turns down
            if not (bullish_alignment and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bearish alignment breaks OR 1d trend turns up
            if not (bearish_alignment and close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals