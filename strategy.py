#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation.
# Uses Williams %R(14) on 6m for overbought/oversold conditions.
# Fades at extreme levels when 1d trend is opposite (mean reversion in range).
# Takes continuation signals when 1d trend aligns and price breaks recent highs/lows.
# Volume confirmation requires 20-period volume spike (1.5x EMA).
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.
# Works in both bull and bear markets via trend-adaptive logic.

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 6m Williams %R(14)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # Neutral when no range
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 20-period volume spike (1.5x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for Williams %R and volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry conditions
            # Oversold reversal in downtrend (mean reversion)
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                williams_r_aligned[i] <= -80 and  # Oversold
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Continuation breakout in uptrend
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  williams_r_aligned[i] >= -20 and  # Overbought (momentum)
                  close[i] > np.max(high[i-5:i]) and  # Break recent high
                  vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry conditions
            # Overbought reversal in uptrend (mean reversion)
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  williams_r_aligned[i] >= -20 and  # Overbought
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            # Continuation breakdown in downtrend
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  williams_r_aligned[i] <= -80 and  # Oversold (momentum)
                  close[i] < np.min(low[i-5:i]) and  # Break recent low
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or stop
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                williams_r_aligned[i] >= -20):  # Overbought or momentum shift
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] < np.min(low[i-5:i])):  # Break recent low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or stop
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                williams_r_aligned[i] <= -80):  # Oversold or momentum shift
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] > np.max(high[i-5:i])):  # Break recent high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals