#!/usr/bin/env python3
# Hypothesis: 6h timeframe with weekly volatility-adjusted breakout and daily trend filter.
# Uses weekly ATR-based breakout levels (mean ± k*ATR) to capture volatility expansion.
# Daily EMA50 trend filter ensures trades align with higher timeframe momentum.
# Volume confirmation filters low-liquidity breakouts. Designed for 50-150 total trades over 4 years.
# Works in both bull and bear markets by capturing volatility breakouts in direction of trend.

name = "6h_VolatilityBreakout_TrendFilter_Volume"
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
    volume = prices['volume'].values
    
    # Calculate weekly ATR(14) for volatility-based breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # True Range calculation for weekly data
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate True Range components
    tr1 = w_high - w_low
    tr2 = np.abs(w_high - np.roll(w_close, 1))
    tr3 = np.abs(w_low - np.roll(w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Weekly ATR(14)
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly mean price (average of weekly OHLC)
    weekly_mean = (w_high + w_low + w_close + np.roll(w_close, 1)) / 4
    weekly_mean = pd.Series(weekly_mean).rolling(window=14, min_periods=14).mean().values
    
    # Breakout levels: weekly mean ± 1.5 * weekly ATR
    upper_breakout = weekly_mean + 1.5 * atr_14_1w
    lower_breakout = weekly_mean - 1.5 * atr_14_1w
    
    # Align weekly levels to 6h timeframe (already delayed by weekly close)
    upper_breakout_aligned = align_htf_to_ltf(prices, df_1w, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_1w, lower_breakout)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume filter: current volume > 1.8x 24-period average volume (4 days of 6h bars)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper level + daily uptrend + volume spike
            if close[i] > upper_breakout_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower level + daily downtrend + volume spike
            elif close[i] < lower_breakout_aligned[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly mean or trend reversal
            if close[i] <= weekly_mean[-1] if not np.isnan(weekly_mean[-1]) else False or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly mean or trend reversal
            if close[i] >= weekly_mean[-1] if not np.isnan(weekly_mean[-1]) else False or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals