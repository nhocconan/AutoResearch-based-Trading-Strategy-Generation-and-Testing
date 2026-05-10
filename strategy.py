#!/usr/bin/env python3
# 1d_Weekly_Pivot_Breakout_Momentum
# Hypothesis: Weekly Pivot Point breakout with 1-day momentum confirmation and volume filter.
# Uses weekly pivot levels (calculated from prior week) for breakout signals,
# confirmed by daily RSI momentum and volume surge. Designed to capture strong
# trending moves while avoiding false breakouts in ranging markets.
# Weekly timeframe provides fewer, higher-quality signals to minimize fee drag.
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns.

name = "1d_Weekly_Pivot_Breakout_Momentum"
timeframe = "1d"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points from previous week's OHLC
    # Using standard formula: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Shift to get previous week's values (avoid look-ahead)
    prev_weekly_high = np.concatenate([[weekly_high[0]], weekly_high[:-1]])
    prev_weekly_low = np.concatenate([[weekly_low[0]], weekly_low[:-1]])
    prev_weekly_close = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    
    # Calculate pivot levels
    pivot_point = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    resistance_1 = 2 * pivot_point - prev_weekly_low
    support_1 = 2 * pivot_point - prev_weekly_high
    
    # Align weekly pivot levels to daily timeframe
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    resistance_1_aligned = align_htf_to_ltf(prices, df_1w, resistance_1)
    support_1_aligned = align_htf_to_ltf(prices, df_1w, support_1)
    
    # Daily RSI for momentum confirmation
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_point_aligned[i]) or np.isnan(resistance_1_aligned[i]) or
            np.isnan(support_1_aligned[i]) or np.isnan(rsi_values[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Weekly R1 with bullish momentum and volume
            if (close[i] > resistance_1_aligned[i] and
                rsi_values[i] > 55 and  # Bullish momentum
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Weekly S1 with bearish momentum and volume
            elif (close[i] < support_1_aligned[i] and
                  rsi_values[i] < 45 and  # Bearish momentum
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly pivot or momentum fails
            if (close[i] < pivot_point_aligned[i] or
                rsi_values[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly pivot or momentum fails
            if (close[i] > pivot_point_aligned[i] or
                rsi_values[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals