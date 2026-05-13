# 1d_WeeklyPivot_R1_S1_Breakout_Trend_Filter_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R1/S1) from 1w data with 1d EMA trend filter and volume confirmation.
# Weekly pivots provide strong support/resistance levels that work in both bull and bear markets.
# The EMA filter ensures trades align with the daily trend, avoiding counter-trend entries.
# Volume confirmation reduces false breakouts. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WeeklyPivot_R1_S1_Breakout_Trend_Filter_Volume"
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

    # Get 1w data for weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_range = df_1w['high'] - df_1w['low']
    camarilla_r1 = typical_price + weekly_range * 1.1 / 12
    camarilla_s1 = typical_price - weekly_range * 1.1 / 12
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1.values)

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above weekly R1 + price above 1d EMA (bullish trend) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below weekly S1 + price below 1d EMA (bearish trend) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below weekly S1 or price below 1d EMA
            if (close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above weekly R1 or price above 1d EMA
            if (close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals