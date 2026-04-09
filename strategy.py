#!/usr/bin/env python3
# 1d_1w_rsi_engulfing_v1
# Hypothesis: Daily RSI(2) extremes with weekly trend filter and bullish/bearish engulfing candle confirmation.
# Long: RSI(2) < 10 + weekly close > weekly open + bullish engulfing candle.
# Short: RSI(2) > 90 + weekly close < weekly open + bearish engulfing candle.
# Exit: RSI(2) crosses above 50 (long) or below 50 (short).
# Works in both bull and bear markets as RSI extremes capture overextended moves.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_engulfing_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend: bullish if close > open
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Calculate daily RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate engulfing candles
    bullish_engulfing = (close > open_) & (open_ > np.roll(close, 1)) & (close > np.roll(open_, 1))
    bearish_engulfing = (close < open_) & (open_ < np.roll(close, 1)) & (close < np.roll(open_, 1))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(weekly_bullish_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50
            if rsi[i] >= 50 and rsi[i-1] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50
            if rsi[i] <= 50 and rsi[i-1] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI(2) < 10 + weekly bullish + bullish engulfing
            if rsi[i] < 10 and weekly_bullish_aligned[i] > 0.5 and bullish_engulfing[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI(2) > 90 + weekly bearish + bearish engulfing
            elif rsi[i] > 90 and weekly_bullish_aligned[i] < 0.5 and bearish_engulfing[i]:
                position = -1
                signals[i] = -0.25
    
    return signals