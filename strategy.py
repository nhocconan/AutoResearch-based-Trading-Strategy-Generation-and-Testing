#!/usr/bin/env python3
# 4h_LinearRegressionChannel_Breakout_1dTrend_Volume
# Hypothesis: Uses linear regression channel (20-period) on 4h for breakout signals.
# Long when price breaks above upper channel with volume > 1.5x average and price > daily EMA50.
# Short when price breaks below lower channel with volume > 1.5x average and price < daily EMA50.
# Exits when price crosses back below/above the 50-period linear regression (midline).
# Designed for 20-40 trades/year to avoid overtrading and work in both bull and bear markets.
# Linear regression adapts to volatility and trend, providing dynamic support/resistance.

name = "4h_LinearRegressionChannel_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Linear regression channel (20-period) on 4h
    def calculate_lr_channel(src, length):
        if len(src) < length:
            return np.full(len(src), np.nan), np.full(len(src), np.nan), np.full(len(src), np.nan)
        
        upper = np.full(len(src), np.nan)
        lower = np.full(len(src), np.nan)
        midline = np.full(len(src), np.nan)
        
        for i in range(length-1, len(src)):
            y = src[i-length+1:i+1]
            x = np.arange(length)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Predictions at start and end of window
            y_start = intercept  # at x=0
            y_end = intercept + slope * (length - 1)  # at x=length-1
            
            # Standard deviation of residuals
            y_pred = intercept + slope * x
            residuals = y - y_pred
            std_residuals = np.std(residuals)
            
            # Channel lines: midline ± 2 * std_residuals
            midline[i] = (y_start + y_end) / 2
            upper[i] = midline[i] + 2 * std_residuals
            lower[i] = midline[i] - 2 * std_residuals
        
        return upper, lower, midline
    
    lr_upper, lr_lower, lr_midline = calculate_lr_channel(close, 20)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(lr_upper[i]) or np.isnan(lr_lower[i]) or np.isnan(lr_midline[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper channel with volume confirmation and uptrend
            if close[i] > lr_upper[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower channel with volume confirmation and downtrend
            elif close[i] < lr_lower[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below midline
            if close[i] < lr_midline[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above midline
            if close[i] > lr_midline[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals