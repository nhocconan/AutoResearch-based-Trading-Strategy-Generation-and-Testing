#!/usr/bin/env python3
# 6h_WilliamsVixFix_MeanReversion
# Hypothesis: The Williams Vix Fix (WVF) indicator identifies oversold/overbought conditions
# by measuring volatility compression and expansion. In mean-reverting markets, extreme WVF
# readings (>0.8) signal potential reversals. Combined with Bollinger Band mean reversion
# (price outside 2.5 SD bands) and 1d trend filter (EMA50), this creates high-probability
# mean reversion trades. Works in both bull and bear markets as it fades extremes rather
# than following trends. Target: 50-150 total trades over 4 years with disciplined exits.

name = "6h_WilliamsVixFix_MeanReversion"
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

    # Calculate Williams Vix Fix (WVF)
    # WVF = ((Highest High in period - Low) / Highest High in period) * 100
    # High values indicate fear/volatility spikes
    lookback = 22
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_high - low) / highest_high) * 100
    wvf = np.where(highest_high == 0, 0, wvf)  # Avoid division by zero

    # Bollinger Bands (20, 2.5) for mean reversion signals
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (std_20 * 2.5)
    lower_bb = sma_20 - (std_20 * 2.5)

    # Get daily data for EMA trend filter (to avoid trading against strong trend)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: avoid low-volume false signals
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(wvf[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: WVF > 0.8 (extreme fear) + price below lower BB + above 1d EMA50 (avoid strong downtrend) + volume filter
            if (wvf[i] > 80 and 
                close[i] < lower_bb[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.2):
                signals[i] = 0.25
                position = 1
            # SHORT: WVF > 0.8 (extreme fear) + price above upper BB + below 1d EMA50 (avoid strong uptrend) + volume filter
            elif (wvf[i] > 80 and 
                  close[i] > upper_bb[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to mean (SMA20) or WVF normalizes (< 30)
            if (close[i] >= sma_20[i] or wvf[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to mean (SMA20) or WVF normalizes (< 30)
            if (close[i] <= sma_20[i] or wvf[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals