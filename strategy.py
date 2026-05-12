#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Regime
# Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R1 from previous 12h with volume >1.5x average and 12h EMA50 trending up; sell when price breaks below Camarilla S1 with volume >1.5x average and 12h EMA50 trending down. Added Choppiness Index (CHOP) > 61.8 regime filter to avoid false breakouts in ranging markets. Targets 20-40 trades per year to reduce fee drag and improve generalization in bull/bear markets. Uses discrete position sizing (0.25) to minimize churn.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Regime"
timeframe = "4h"
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

    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R1 = close + (high - low) * 1.12/12, S1 = close - (high - low) * 1.12/12
    range_12h = high_12h - low_12h
    camarilla_r1 = close_12h + range_12h * 1.12 / 12
    camarilla_s1 = close_12h - range_12h * 1.12 / 12

    # Use previous 12h bar's levels (shift by 1)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_prev)

    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # 12h Choppiness Index (CHOP) for regime filter
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))

    tr = true_range(high_12h, low_12h, np.roll(close_12h, 1))
    tr[0] = 0  # first value has no previous close

    atr_period = 14
    tr_sum = np.zeros_like(tr)
    tr_sum[atr_period-1] = np.nansum(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / atr_period) + tr[i]
    atr = tr_sum / atr_period

    max_high = np.zeros_like(high_12h)
    min_low = np.zeros_like(low_12h)
    max_high[atr_period-1] = np.max(high_12h[:atr_period])
    min_low[atr_period-1] = np.min(low_12h[:atr_period])
    for i in range(atr_period, len(high_12h)):
        max_high[i] = max(max_high[i-1], high_12h[i])
        min_low[i] = min(min_low[i-1], low_12h[i])

    # Avoid division by zero
    range_max_min = max_high - min_low
    chop = np.where(range_max_min != 0, 100 * np.log10(atr * atr_period / range_max_min) / np.log10(atr_period), 50)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)

    # Volume confirmation: volume > 1.5x 20-period average (approx 10 hours)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 12h uptrend + volume spike + CHOP > 61.8 (trending regime)
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 12h downtrend + volume spike + CHOP > 61.8 (trending regime)
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR trend turns down OR CHOP < 38.2 (range regime)
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema50_12h_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR trend turns up OR CHOP < 38.2 (range regime)
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema50_12h_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals