# The Alligator is a trend-following indicator that uses three smoothed moving averages (jaws, teeth, lips) to identify the presence and direction of a trend. When the lines are intertwined, the market is sleeping (range-bound). When they diverge in order (lips > teeth > jaws for uptrend, lips < teeth < jaws for downtrend), a trend is present. This strategy uses the Alligator on the daily timeframe to determine trend direction and entry timing on the 12h chart, combined with volume confirmation to avoid false breakouts. Designed for low trade frequency to minimize fee drag in the 12h timeframe.

name = "12h_Alligator_Trend_Volume"
timeframe = "12h"
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

    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Alligator lines (Smoothed Moving Average with period 5, 8, 13)
    # Jaws: SMA(13) smoothed by 8 periods
    # Teeth: SMA(8) smoothed by 5 periods
    # Lips: SMA(5) smoothed by 3 periods
    # Using median price (HL/2) as input
    median_price_1d = (high_1d + low_1d) / 2
    median_series = pd.Series(median_price_1d)

    # Jaws: 13-period SMA, then 8-period smoothing
    jaws_raw = median_series.rolling(window=13, min_periods=13).mean()
    jaws = jaws_raw.rolling(window=8, min_periods=8).mean()

    # Teeth: 8-period SMA, then 5-period smoothing
    teeth_raw = median_series.rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean()

    # Lips: 5-period SMA, then 3-period smoothing
    lips_raw = median_series.rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean()

    jaws_values = jaws.values
    teeth_values = teeth.values
    lips_values = lips.values

    # Align Alligator lines to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_values)

    # Volume confirmation: 2.0x 20-period SMA on 12h
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Alligator needs 13 bars for lips calculation
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaws (bullish alignment) with volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (bearish alignment) with volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: When Alligator lines re-intertwine (lips < jaws)
            if lips_aligned[i] < jaws_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: When Alligator lines re-intertwine (lips > jaws)
            if lips_aligned[i] > jaws_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals