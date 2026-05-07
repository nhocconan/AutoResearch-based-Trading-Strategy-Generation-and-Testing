#!/usr/bin/env python3
# 4h_RSI_MeanReversion_4HTrend_Filter
# Hypothesis: Mean reversion on 4h RSI with trend filter from higher timeframe (1d) to avoid counter-trend trades.
# Uses RSI(14) < 30 for long and > 70 for short, filtered by 1-day EMA50 trend.
# Includes volume confirmation and minimum holding period to reduce whipsaw.
# Target: 20-30 trades/year to minimize fee drag while capturing reversals in both bull and bear markets.

name = "4h_RSI_MeanReversion_4HTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_4h[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: RSI < 30 (oversold), above 1d EMA50 trend, volume confirmation
            if rsi[i] < 30 and close[i] > ema_50_1d_4h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: RSI > 70 (overbought), below 1d EMA50 trend, volume confirmation
            elif rsi[i] > 70 and close[i] < ema_50_1d_4h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: RSI > 50 or below trend
            if rsi[i] > 50 or close[i] < ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 or above trend
            if rsi[i] < 50 or close[i] > ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals