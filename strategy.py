#!/usr/bin/env python3
# 6h_WilliamsVixFix_MeanReversion_1dTrend
# Hypothesis: Williams Vix Fix (WVF) identifies extreme oversold/overbought conditions by measuring volatility relative to recent highs.
# In both bull and bear markets, reversals from these extremes tend to revert to the mean, especially when aligned with the daily trend.
# Uses 1d trend filter to avoid counter-trend trades, improving win rate. Designed for low frequency (~15-35 trades/year) to minimize fee drag.
# Williams Vix Fix formula: wvf = ((highest_high - low) / highest_high) * 100, where highest_high is the highest high over lookback period.
# Low WVF values indicate high volatility (fear), suggesting potential mean reversion upward.
# High WVF values indicate low volatility (complacency), suggesting potential mean reversion downward.

name = "6h_WilliamsVixFix_MeanReversion_1dTrend"
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
    
    # Williams Vix Fix parameters
    wvf_period = 22  # lookback period for highest high
    
    # Calculate highest high over the lookback period
    high_series = pd.Series(high)
    highest_high = high_series.rolling(window=wvf_period, min_periods=wvf_period).max().values
    
    # Williams Vix Fix: measures volatility as percentage deviation from recent high
    # Higher values = lower volatility (complacency), Lower values = higher volatility (fear)
    wvf_raw = ((highest_high - low) / highest_high) * 100
    
    # Daily trend filter: EMA34 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily trend to 6h timeframe
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volatility regime filter: use WVF itself to identify extreme readings
    # We'll use the 50th percentile of WVF as dynamic threshold
    wvf_series = pd.Series(wvf_raw)
    wvf_median = wvf_series.rolling(window=50, min_periods=50).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, wvf_period)  # ensure we have WVF and median
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(wvf_raw[i]) or 
            np.isnan(wvf_median[i]) or np.isnan(trend_1d_up_aligned[i]) or 
            np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: extreme fear (low WVF = high volatility) during daily uptrend
            if (wvf_raw[i] < wvf_median[i] * 0.7 and  # Oversold: volatility spike
                trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Enter short: extreme complacency (high WVF = low volatility) during daily downtrend
            elif (wvf_raw[i] > wvf_median[i] * 1.3 and  # Overbought: low volatility
                  trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility normalization or trend failure
            if (wvf_raw[i] > wvf_median[i] * 0.9 or  # Volatility normalized
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility normalization or trend failure
            if (wvf_raw[i] < wvf_median[i] * 1.1 or  # Volatility normalized
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals