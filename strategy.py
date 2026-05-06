#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams %R with Bollinger Band squeeze filter
# Long when %R crosses above -80 (oversold exit) with Bollinger Band width < 20th percentile (low volatility)
# Short when %R crosses below -20 (overbought entry) with Bollinger Band width < 20th percentile
# Uses daily Williams %R for mean reversion signals and Bollinger Band width for volatility regime filter
# Designed to work in both bull and bear markets by capturing mean reversion in low volatility regimes
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dWilliamsR_BollingerSqueeze_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # Handle division by zero
    
    # Calculate 1-day Bollinger Band Width (20-period, 2 std dev)
    sma_20 = df_1d['close'].rolling(window=20, min_periods=20).mean()
    std_20 = df_1d['close'].rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = ((upper_bb - lower_bb) / sma_20) * 100  # Percentage width
    bb_width = bb_width.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Bollinger Band squeeze: width < 20th percentile (low volatility regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile_20 = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20)
    bb_squeeze = bb_width < bb_width_percentile_20.values
    
    # Align indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Williams %R signal generation: cross above -80 for long, cross below -20 for short
    williams_r_series = pd.Series(williams_r_aligned)
    williams_r_prev = williams_r_series.shift(1).fillna(50).values
    
    long_signal = (williams_r_aligned > -80) & (williams_r_prev <= -80)  # Cross above -80
    short_signal = (williams_r_aligned < -20) & (williams_r_prev >= -20)  # Cross below -20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous value
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(bb_squeeze_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 (exiting oversold) with Bollinger squeeze
            if long_signal[i] and bb_squeeze_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 (entering overbought) with Bollinger squeeze
            elif short_signal[i] and bb_squeeze_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum weakening)
            if williams_r_aligned[i] < -50 and williams_r_prev > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum weakening)
            if williams_r_aligned[i] > -50 and williams_r_prev < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals