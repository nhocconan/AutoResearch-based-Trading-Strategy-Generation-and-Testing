#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels from daily data identify key support/resistance. Breakouts above R3 or below S3 with volume confirmation and aligned daily trend (price above/below daily EMA34) capture strong moves. Works in bull via upside breakouts and bear via downside breakdowns. Target: 15-30 trades/year.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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

    # Calculate Camarilla pivot levels from daily data (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Camarilla: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we need previous day's values, so shift by 1
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate levels
    rang = prev_high - prev_low
    r3 = prev_close + 1.1 * rang
    s3 = prev_close - 1.1 * rang
    r4 = prev_close + 1.5 * rang
    s4 = prev_close - 1.5 * rang
    
    # Align to 6h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)

    # Daily trend filter: EMA34 on daily close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # warmup period
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R3 with volume spike and daily uptrend (close > EMA34)
            if (close[i] > r3_aligned[i] and 
                volume[i] > vol_avg_20[i] * 2.0 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and daily downtrend (close < EMA34)
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 2.0 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or daily trend turns down
            if (close[i] < r3_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or daily trend turns up
            if (close[i] > s3_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals