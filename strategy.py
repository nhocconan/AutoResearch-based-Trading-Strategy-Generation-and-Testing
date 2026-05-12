#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use 1d Camarilla pivot levels (R1, S1) with 12h breakout logic. 
# Enter long when price breaks above R1 with volume confirmation and 1d uptrend.
# Enter short when price breaks below S1 with volume confirmation and 1d downtrend.
# Exit when price returns to the mean (PP) or reverses. 
# Designed for 12-25 trades/year per symbol, works in both bull and bear via mean-reversion structure.
# Uses volume confirmation to avoid false breakouts and Camarilla levels for institutional relevance.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate 1d Camarilla pivot levels: based on previous day's high, low, close
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    # We use R1, S1, and PP (pivot point)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values

    # Calculate Pivot Point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Calculate R1 and S1
    r1 = prev_close + ((prev_high - prev_low) * 1.0833)
    s1 = prev_close - ((prev_high - prev_low) * 1.0833)

    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1d EMA34 trend filter (smoothed trend)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (2 days)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]

        if position == 0:
            # LONG: Price breaks above R1 with volume and uptrend
            if close[i] > r1_aligned[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and downtrend
            elif close[i] < s1_aligned[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (mean reversion) or breaks S1
            if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (mean reversion) or breaks R1
            if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals