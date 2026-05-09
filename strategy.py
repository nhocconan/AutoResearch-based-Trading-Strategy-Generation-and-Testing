#!/usr/bin/env python3
# 6h_Weekly_Trend_Daily_Momentum
# Hypothesis: Combines weekly trend direction (from weekly close > weekly SMA50) with daily momentum (RSI > 55) on 6-hour timeframe.
# Weekly trend filter ensures we trade with the higher timeframe trend, while daily RSI provides entry timing.
# Works in both bull and bear markets by following the weekly trend direction. Target: 15-25 trades/year per symbol.

name = "6h_Weekly_Trend_Daily_Momentum"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA50
    sma_50 = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        sma_50[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            sma_50[i] = (sma_50[i-1] * 49 + close_1w[i]) / 50
    
    # Weekly trend: 1 if close > SMA50, -1 if close < SMA50
    weekly_trend = np.full_like(close_1w, 0)
    valid_sma = ~np.isnan(sma_50)
    weekly_trend[valid_sma & (close_1w > sma_50)] = 1
    weekly_trend[valid_sma & (close_1w < sma_50)] = -1
    
    # Align weekly trend to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Get daily data for momentum (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 50)
    # For first 14 values, RSI is undefined (set to 50 neutral)
    rsi[:14] = 50
    
    # Align daily RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 1)  # Start from second bar to allow for previous bar comparison
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_trend_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_val = weekly_trend_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: Weekly uptrend AND daily RSI > 55 (bullish momentum)
            if weekly_trend_val == 1 and rsi_val > 55:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly downtrend AND daily RSI < 45 (bearish momentum)
            elif weekly_trend_val == -1 and rsi_val < 45:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend turns down OR RSI falls below 50 (momentum fade)
            if weekly_trend_val == -1 or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend turns up OR RSI rises above 50 (momentum fade)
            if weekly_trend_val == 1 or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals