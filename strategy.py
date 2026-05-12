#!/usr/bin/env python3
# 6h_ChoppinessIndex_Regime_ADX_Trend
# Hypothesis: Uses Choppiness Index (14) to detect market regime (trend vs range) and ADX (14) to confirm trend strength.
# In trending regime (CHOP < 38.2) with strong ADX (>25), follow price direction relative to 50-period EMA.
# In ranging regime (CHOP > 61.8), fade moves to Bollinger Band (20,2) extremes.
# Weekly trend filter (EMA50) ensures alignment with higher timeframe trend.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.

name = "6h_ChoppinessIndex_Regime_ADX_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Get daily data for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate indicators on daily data
    close_1d_series = pd.Series(close_1d)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)

    # Choppiness Index (14)
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum()
    max_high_14d = high_1d_series.rolling(window=14, min_periods=14).max()
    min_low_14d = low_1d_series.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_1d / (max_high_14d - min_low_14d)) / np.log10(14)
    chop_values = chop.values

    # ADX (14)
    plus_dm = high_1d_series.diff()
    minus_dm = low_1d_series.diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high_1d_series - low_1d_series
    tr2 = (high_1d_series - close_1d_series.shift()).abs()
    tr3 = (low_1d_series - close_1d_series.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr_14)
    minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr_14)
    dx = (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values

    # Bollinger Bands (20,2) for mean reversion in ranging markets
    sma_20 = close_1d_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d_series.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_values = upper_bb.values
    lower_bb_values = lower_bb.values

    # Weekly EMA50 for trend filter
    ema50_1w = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values  # Temporary, will be replaced

    # Actually compute weekly EMA50 from weekly data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Align HTF indicators to 6s timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_values)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup period
        # Get aligned values for current 6h bar
        chop_val = chop_aligned[i]
        adx_val = adx_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        ema50_aligned = ema50_1w_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(chop_val) or np.isnan(adx_val) or 
            np.isnan(upper_bb_val) or np.isnan(lower_bb_val) or 
            np.isnan(ema50_aligned)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Determine market regime
            if chop_val < 38.2 and adx_val > 25:  # Trending regime
                # Follow trend: long if price above EMA50, short if below
                if close[i] > ema50_aligned:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema50_aligned:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif chop_val > 61.8:  # Ranging regime
                # Mean revert at Bollinger Band extremes
                if close[i] <= lower_bb_val:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb_val:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # Transition zone, no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long conditions
            if chop_val < 38.2 and adx_val > 25:  # Still in trending regime
                if close[i] < ema50_aligned:  # Trend turned down
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_val > 61.8:  # In ranging regime, exit at mean
                if close[i] >= (upper_bb_val + lower_bb_val) / 2:  # Return to midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition zone, exit
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Exit short conditions
            if chop_val < 38.2 and adx_val > 25:  # Still in trending regime
                if close[i] > ema50_aligned:  # Trend turned up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_val > 61.8:  # In ranging regime, exit at mean
                if close[i] <= (upper_bb_val + lower_bb_val) / 2:  # Return to midpoint
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition zone, exit
                signals[i] = 0.0
                position = 0

    return signals