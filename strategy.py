#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Filter
# Hypothesis: Trade 12-hour breakouts from Camarilla R1/S1 levels during 1-week uptrend/downtrend with volume confirmation.
# Long when price breaks above R1 during 1w uptrend with volume spike. Short when price breaks below S1 during 1w downtrend with volume spike.
# Exit on opposite Camarilla level touch or trend reversal.
# Uses 1w trend filter to avoid counter-trend whipsaws, targeting 15-35 trades/year per symbol.
# Designed to work in both bull (trend-following breakouts) and bear (trend-following breakdowns) markets.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Filter"
timeframe = "12h"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Calculate Camarilla levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    # Camarilla R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_S1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1.values)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1.values)

    # Volume filter: >1.5x 24-period average (2 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_avg_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 in 1w uptrend + volume spike
            if close[i] > camarilla_R1_aligned[i]:
                if close[i] > ema_34_1w_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S1 in 1w downtrend + volume spike
            elif close[i] < camarilla_S1_aligned[i]:
                if close[i] < ema_34_1w_aligned[i] and volume[i] > vol_avg_24[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches S1 or trend turns down
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1w_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches R1 or trend turns up
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1w_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals