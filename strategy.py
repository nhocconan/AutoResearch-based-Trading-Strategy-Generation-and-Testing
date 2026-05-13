#!/usr/bin/env python3
# 12h_RVOL_MeanReversion_1dFilter
# Hypothesis: Mean reversion on high relative volume (RVOL) spikes at 12h timeframe, filtered by 1d trend.
# Long when price drops >1.5 sigma on RVOL>2.0 during 1d uptrend. Short when price rises >1.5 sigma on RVOL>2.0 during 1d downtrend.
# Uses RVOL to detect exhaustion moves and 1d trend to avoid counter-trend trades.
# Designed for low trade frequency (~20-40/year) to minimize fee drag in ranging/bear markets.

name = "12h_RVOL_MeanReversion_1dFilter"
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

    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first tr is undefined
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)

    # 20-period RVOL (volume / 20-period average volume)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / np.where(vol_avg_20 > 0, vol_avg_20, np.inf)

    # 20-period price z-score (deviation from mean in units of 1d ATR)
    price_mean_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    price_std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    # Normalize by 1d ATR to make it volatility-adjusted
    zscore = (close - price_mean_20) / np.where(price_std_20 > 0, price_std_20, np.inf) * np.where(atr_14_1d_aligned > 0, atr_14_1d_aligned, 1.0)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN or invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rvol[i]) or np.isnan(zscore[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price oversold (z-score < -1.5) on high volume (RVOL>2.0) during 1d uptrend
            if zscore[i] < -1.5 and rvol[i] > 2.0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price overbought (z-score > 1.5) on high volume (RVOL>2.0) during 1d downtrend
            elif zscore[i] > 1.5 and rvol[i] > 2.0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to mean (z-score > -0.5) or trend turns down
            if zscore[i] > -0.5 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to mean (z-score < 0.5) or trend turns up
            if zscore[i] < 0.5 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals