#!/usr/bin/env python3
name = "1d_LinearRegTrend_20period_1wTrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Linear Regression Slope (20-period) - trend direction
    # Manual calculation to avoid look-ahead
    period = 20
    sum_x = period * (period - 1) // 2  # 0+1+...+(period-1)
    sum_x2 = (period - 1) * period * (2 * period - 1) // 6  # sum of squares
    
    linreg_slope = np.full(n, np.nan)
    for i in range(period - 1, n):
        y_vals = close[i - period + 1:i + 1]
        sum_y = np.sum(y_vals)
        sum_xy = np.sum(y_vals * np.arange(period))
        # slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x*sum_x)
        numerator = period * sum_xy - sum_x * sum_y
        denominator = period * sum_x2 - sum_x * sum_x
        if denominator != 0:
            linreg_slope[i] = numerator / denominator
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # Volume 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(linreg_slope[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Positive linear regression slope (uptrend) + price above weekly EMA50 + volume confirmation
            if (linreg_slope[i] > 0 and 
                close[i] > ema50_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Negative linear regression slope (downtrend) + price below weekly EMA50 + volume confirmation
            elif (linreg_slope[i] < 0 and 
                  close[i] < ema50_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend turns negative OR price breaks below weekly EMA50
            if linreg_slope[i] <= 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend turns positive OR price breaks above weekly EMA50
            if linreg_slope[i] >= 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals