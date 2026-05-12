#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_Volume
# Hypothesis: Daily Camarilla pivot levels (R1/S1) act as strong support/resistance. Breakouts above R1 or below S1 with volume confirmation and weekly EMA trend filter provide high-probability trades. Weekly EMA34 filters for long-term trend direction, reducing whipsaws in ranging markets. Designed for 1d timeframe to target 7-25 trades per year, minimizing fee drag while capturing major moves in both bull and bear markets.

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_Volume"
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
    volume = prices['volume'].values

    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate Camarilla levels for daily data (using previous day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first values to avoid roll issues (use first available data)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12

    # Get weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: 2x 20-period SMA (higher threshold to reduce trades)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to have valid previous day data
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 with volume confirmation and weekly uptrend
            if (close[i] > r1[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 with volume confirmation and weekly downtrend
            elif (close[i] < s1[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close drops below S1 (reversal signal)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises above R1 (reversal signal)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals