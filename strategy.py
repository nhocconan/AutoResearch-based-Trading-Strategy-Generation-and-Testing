#!/usr/bin/env python3
"""
1d_Williams_Alligator_1w_Trend_Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 1d identifies trend direction. 
1w EMA34 confirms higher timeframe trend. Long when Lips > Teeth > Jaw and price > 1w EMA34; 
Short when Lips < Teeth < Jaw and price < 1w EMA34. Filters reduce false signals in ranging markets.
Designed for 15-30 trades/year to minimize fee drag while capturing sustained trends in bull and bear markets.
"""

name = "1d_Williams_Alligator_1w_Trend_Filter"
timeframe = "1d"
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

    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate Williams Alligator (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    jaw = smma(close, 13)  # Jaw (13-period SMMA)
    teeth = smma(close, 8)  # Teeth (8-period SMMA)
    lips = smma(close, 5)   # Lips (5-period SMMA)

    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)

    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any values are NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_34_1w_aligned[i]
        close_val = close[i]

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34
            if lips_val > teeth_val > jaw_val and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34
            elif lips_val < teeth_val < jaw_val and close_val < ema_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish alignment (Lips < Teeth < Jaw) OR price < 1w EMA34
            if lips_val < teeth_val < jaw_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish alignment (Lips > Teeth > Jaw) OR price > 1w EMA34
            if lips_val > teeth_val > jaw_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals