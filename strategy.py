#!/usr/bin/env python3
"""
1d_ElderRay_1wTrend_Pullback
Hypothesis: On daily timeframe, Elder Ray Index (bull/bear power) with 13-day EMA 
identifies trend from weekly trend filter. Long when weekly EMA50 uptrend, 
daily bear power crosses above zero (bulls gaining control) and price pulls back 
to EMA13. Short when weekly downtrend, daily bull power crosses below zero 
(bears gaining control) and price rallies to EMA13. Uses volume confirmation 
to avoid false signals. Targets 15-25 trades/year (60-100 total over 4 years).
Works in bull via trend continuation and bear via counter-trend pulls at 
weekly trend extremes.
"""

name = "1d_ElderRay_1wTrend_Pullback"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Daily EMA13 for pullback entries
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Daily EMA13 for Elder Ray calculation
    ema13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_close
    bear_power = low - ema13_close

    # Volume confirmation: 1.3x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current daily bar
        weekly_trend = ema50_1w_aligned[i]  # >0 = uptrend, <0 = downtrend (we'll use slope)
        # Actually use the EMA value directly for trend: price above/below EMA50
        weekly_ema50 = ema50_1w_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        ema13_val = ema13[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(weekly_ema50) or np.isnan(bp) or 
            np.isnan(br) or np.isnan(ema13_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend (price above weekly EMA50), 
            #       Bull power turns positive (bulls in control),
            #       Price pulls back to EMA13 (or slightly below)
            if (close[i] > weekly_ema50 and      # Weekly uptrend filter
                bp > 0 and                       # Bull power positive
                bp < bp[i-1] and                 # Bull power declining (pullback)
                close[i] <= ema13_val * 1.005 and # Near EMA13 (within 0.5% above)
                volume[i] > vol_avg_val * 1.3):   # Volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend (price below weekly EMA50),
            #        Bear power turns negative (bears in control),
            #        Price rallies to EMA13 (or slightly above)
            elif (close[i] < weekly_ema50 and     # Weekly downtrend filter
                  br < 0 and                      # Bear power negative
                  br > br[i-1] and                # Bear power declining (less negative = rally)
                  close[i] >= ema13_val * 0.995 and # Near EMA13 (within 0.5% below)
                  volume[i] > vol_avg_val * 1.3):   # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns down OR bear power turns negative
            if (close[i] < weekly_ema50 or br < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns up OR bull power turns positive
            if (close[i] > weekly_ema50 or bp > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals