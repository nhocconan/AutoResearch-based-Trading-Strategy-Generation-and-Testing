#!/usr/bin/env python3
"""
6h_LinearRegReversal_1dTrend_Volume
Hypothesis: On 6h timeframe, mean-reversion to linear regression trend from 1d provides edge in both bull and bear markets.
Long when price deviates below 1.5σ of 60-period linear regression AND 1d trend is up (EMA34 rising) AND volume > 1.5x average.
Short when price deviates above +1.5σ AND 1d trend is down AND volume surge.
Uses 1d Bollinger Band width < 50th percentile to avoid choppy regimes.
Targets 12-37 trades/year (50-150 total over 4 years) with low turnover to minimize fee drag.
Works in bull via pullbacks to rising trend and bear via bounces off declining trend.
"""

name = "6h_LinearRegReversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = np.diff(ema34_1d, prepend=ema34_1d[0])  # slope of EMA
    ema34_1d_slope = np.append(ema34_1d_slope, ema34_1d_slope[-1])  # match length
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)

    # Calculate 1d Bollinger Band width (20, 2) for squeeze filter
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    # Percentile rank of bb_width over lookback
    bb_width_rank = pd.Series(bb_width_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1d, bb_width_rank)

    # Calculate 60-period linear regression on 6h close
    def linreg_slope_intercept(arr, window):
        n = len(arr)
        slope = np.full(n, np.nan)
        intercept = np.full(n, np.nan)
        for i in range(window-1, n):
            y = arr[i-window+1:i+1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                continue
            # Use only non-nan values for regression
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            A = np.vstack([x_valid, np.ones(len(x_valid))]).T
            m, c = np.linalg.lstsq(A, y_valid, rcond=None)[0]
            slope[i] = m
            intercept[i] = c
        return slope, intercept

    slope_60, intercept_60 = linreg_slope_intercept(close, 60)
    # Predicted value from linear regression
    pred_60 = slope_60 * np.arange(len(close)) + intercept_60
    # Residuals (actual - predicted)
    residuals = close - pred_60
    # Standard deviation of residuals
    std_resid = pd.Series(residuals).rolling(window=60, min_periods=30).std().values
    # Z-score of residuals
    zscore = residuals / std_resid

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Get aligned values for current 6h bar
        ema34_slope = ema34_1d_slope_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        z = zscore[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34_slope) or np.isnan(bb_rank) or 
            np.isnan(z) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Squeeze filter: only trade when BB width is in lower 50% (contraction)
        if bb_rank > 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below -1.5σ + 1d trend up (slope > 0) + volume surge
            if (z < -1.5 and 
                ema34_slope > 0 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above +1.5σ + 1d trend down (slope < 0) + volume surge
            elif (z > 1.5 and 
                  ema34_slope < 0 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above linear regression or trend turns down
            if (z > 0 or ema34_slope < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below linear regression or trend turns up
            if (z < 0 or ema34_slope > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals