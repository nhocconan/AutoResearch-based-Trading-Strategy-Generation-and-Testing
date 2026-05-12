#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: On 4h timeframe, buy when price touches Camarilla S1 level with 1d uptrend and volume spike; sell when price touches Camarilla R1 level with 1d downtrend and volume spike. Uses Camarilla pivot levels from daily timeframe for mean reversion in ranging markets, filtered by 1d EMA trend and volume confirmation to avoid false signals. Targets 20-40 trades per year to minimize fee drag while capturing high-probability reversals.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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

    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for previous day
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    #          S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+C)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d

    # Calculate levels for previous day (to avoid look-ahead)
    typical_price_prev = np.roll(typical_price_1d, 1)
    range_prev = np.roll(range_1d, 1)
    typical_price_prev[0] = np.nan
    range_prev[0] = np.nan

    # Camarilla levels
    R1 = typical_price_prev + (range_prev * 1.1 / 12)
    S1 = typical_price_prev - (range_prev * 1.1 / 12)

    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2.0x 24-period average (approx 6 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches S1 + 1d uptrend + volume spike
            if (low[i] <= S1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 + 1d downtrend + volume spike
            elif (high[i] >= R1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above typical price (mean reversion complete)
            if close[i] >= typical_price_prev[np.searchsorted(df_1d.index, prices.index[i]) if i < len(prices) else -1]:
                # Simplified exit: price reaches midpoint between S1 and R1
                midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
                if close[i] >= midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below typical price
            midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals