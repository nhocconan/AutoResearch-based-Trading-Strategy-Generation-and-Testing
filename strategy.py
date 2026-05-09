#!/usr/bin/env python3
# 4h_AngleBased_Trend_Scalp
# Hypothesis: Uses 4h price angle (slope of 10-bar linear regression) to detect strong momentum bursts,
# confirmed by volume >2x 20-bar average and filtered by 1d EMA50 trend direction.
# Works in bull/bear by only taking trades in direction of higher timeframe trend.
# Target: 20-40 trades/year on 4h timeframe.

name = "4h_AngleBased_Trend_Scalp"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h price angle (slope of 10-bar linear regression) - last 10 bars
    def linreg_slope(y, x=None):
        if x is None:
            x = np.arange(len(y))
        n_reg = len(y)
        if n_reg < 2:
            return np.nan
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        denominator = n_reg * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return np.nan
        return (n_reg * sum_xy - sum_x * sum_y) / denominator
    
    price_slope = np.full(n, np.nan)
    lookback = 10
    for i in range(lookback - 1, n):
        y_segment = close[i - lookback + 1:i + 1]
        slope = linreg_slope(y_segment)
        price_slope[i] = slope
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, lookback - 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(price_slope[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Strong upward price angle + volume confirmation + bullish 1d trend
            if price_slope[i] > 0.1 and volume_ratio[i] > 2.0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Strong downward price angle + volume confirmation + bearish 1d trend
            elif price_slope[i] < -0.1 and volume_ratio[i] > 2.0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price angle turns negative or trend turns bearish
            if price_slope[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price angle turns positive or trend turns bullish
            if price_slope[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals