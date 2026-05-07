#!/usr/bin/env python3
# 12h_WilsonBreakout_1wTrend_VolumeFilter
# Hypothesis: Use weekly trend (price above/below 200 EMA) to filter 12h Wilson (breakout of 12h high/low over 20 periods) with volume confirmation (1.5x 50-period volume average). Target: 15-30 trades/year per symbol. Wilson channels adapt to volatility, providing dynamic support/resistance. Works in bull (breakouts with trend) and bear (fades from extremes with volume). Weekly trend filter reduces whipsaws.

name = "12h_WilsonBreakout_1wTrend_VolumeFilter"
timeframe = "12h"
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
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly 200 EMA for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Wilson channels: 20-period high/low on 12h
    period = 20
    # Calculate rolling max/min using pandas for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    wilson_high = high_series.rolling(window=period, min_periods=period).max().values
    wilson_low = low_series.rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation: 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 50)  # Ensure we have weekly EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(wilson_high[i]) or 
            np.isnan(wilson_low[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Wilson high, above weekly EMA200 (uptrend), volume spike
            if (close[i] > wilson_high[i] and 
                close[i] > ema200_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Wilson low, below weekly EMA200 (downtrend), volume spike
            elif (close[i] < wilson_low[i] and 
                  close[i] < ema200_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below Wilson low (opposite boundary)
            if close[i] <= wilson_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above Wilson high (opposite boundary)
            if close[i] >= wilson_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals