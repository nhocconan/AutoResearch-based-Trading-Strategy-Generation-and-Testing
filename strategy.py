#!/usr/bin/env python3
# 6h_12h_1d_1w_EhlersFisherTransform_Trend_Filter
# Hypothesis: Uses Ehlers Fisher Transform on 1d prices to detect extreme reversals.
# Enters long when Fisher crosses above -1.5 (oversold) with 12h uptrend and volume spike.
# Enters short when Fisher crosses below +1.5 (overbought) with 12h downtrend and volume spike.
# Uses 12h EMA50 as trend filter and 1w EMA200 as regime filter to avoid counter-trend trades.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 12h trend while using Fisher reversals for precise entries.

name = "6h_12h_1d_1w_EhlersFisherTransform_Trend_Filter"
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
    
    # Volume spike: >1.5x 20-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Ehlers Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ehlers Fisher Transform (9-period)
    # Normalize price to [-1, 1] range over lookback period
    def normalize(price, low, high):
        range_val = high - low
        # Avoid division by zero
        range_val = np.where(range_val == 0, 1e-10, range_val)
        return 2 * ((price - low) / range_val) - 1
    
    # Calculate median price for normalization
    median_price = (high_1d + low_1d) / 2
    
    # Normalize over 9-period lookback
    period = 9
    price_series = pd.Series(median_price)
    low_series = pd.Series(low_1d)
    high_series = pd.Series(high_1d)
    
    # Rolling min/max for normalization
    roll_low = low_series.rolling(window=period, min_periods=period).min().values
    roll_high = high_series.rolling(window=period, min_periods=period).max().values
    
    # Normalized price
    normalized = normalize(median_price, roll_low, roll_high)
    
    # Apply smoothing and Fisher transform
    # Smooth normalized price with 2-period EMA
    smoothed = pd.Series(normalized).ewm(span=2, adjust=False).mean().values
    
    # Fisher transform: 0.5 * ln((1+smoothed)/(1-smoothed))
    # Clip smoothed to avoid domain errors in log
    smoothed_clipped = np.clip(smoothed, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + smoothed_clipped) / (1 - smoothed_clipped))
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w EMA200 for regime filter (avoid counter-trend in strong trends)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 6h timeframe
    fisher_aligned = align_htf_to_ltf(prices, df_1d, fisher)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(fisher_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Fisher crosses above -1.5 (oversold reversal) + 12h EMA50 uptrend + volume spike
            # Plus regime filter: price above 1w EMA200 in uptrend, below in downtrend
            if (fisher[i] > -1.5 and fisher[i-1] <= -1.5 and  # crossover above -1.5
                close[i] > ema_50_12h_aligned[i] and
                close[i] > ema_200_1w_aligned[i] and  # bullish regime: above weekly EMA200
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 (overbought reversal) + 12h EMA50 downtrend + volume spike
            # Plus regime filter: price below 1w EMA200 in downtrend
            elif (fisher[i] < 1.5 and fisher[i-1] >= 1.5 and  # crossover below +1.5
                  close[i] < ema_50_12h_aligned[i] and
                  close[i] < ema_200_1w_aligned[i] and  # bearish regime: below weekly EMA200
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 (overbought) OR closes below 12h EMA50
            if (fisher[i] < 1.5 and fisher[i-1] >= 1.5) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 (oversold) OR closes above 12h EMA50
            if (fisher[i] > -1.5 and fisher[i-1] <= -1.5) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals