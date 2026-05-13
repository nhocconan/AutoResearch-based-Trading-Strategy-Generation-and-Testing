#!/usr/bin/env python3
# 6h_Keltner_Breakout_1dTrend_Volume
# Hypothesis: Trade Keltner channel breakouts on 6h timeframe filtered by 1d EMA trend and volume spike.
# Long when price breaks above upper Keltner band (EMA + 2*ATR) during 1d uptrend with volume > 2x average.
# Short when price breaks below lower Keltner band (EMA - 2*ATR) during 1d downtrend with volume > 2x average.
# Exit when price crosses back through the EMA (middle band) or trend reverses.
# Designed to capture strong trending moves with volatility expansion, avoiding choppy markets.
# Works in bull markets (trend continuation) and bear markets (trend continuation shorts).

name = "6h_Keltner_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend filter and Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA20 for middle band and trend
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 1d ATR(10) for band width
    tr_1d = np.maximum(np.maximum(
        df_1d['high'] - df_1d['low'],
        np.abs(df_1d['high'] - df_1d['close'].shift(1)),
        np.abs(df_1d['low'] - df_1d['close'].shift(1))
    )).values
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Keltner bands
    upper_1d = ema_20_1d + 2 * atr_10_1d
    lower_1d = ema_20_1d - 2 * atr_10_1d
    
    # Align to 6h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner band in 1d uptrend with volume spike
            if close[i] > upper_1d_aligned[i] and close[i] > ema_20_1d_aligned[i]:
                if volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below lower Keltner band in 1d downtrend with volume spike
            elif close[i] < lower_1d_aligned[i] and close[i] < ema_20_1d_aligned[i]:
                if volume[i] > vol_avg_20[i] * 2.0:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA (middle band) or trend turns down
            if close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA (middle band) or trend turns up
            if close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals