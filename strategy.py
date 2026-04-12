#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_momentum_divergence_v1
# Uses RSI(14) divergence on 12h timeframe with 1d trend filter to catch exhaustion moves.
# Bullish divergence: price makes lower low, RSI makes higher low → long
# Bearish divergence: price makes higher high, RSI makes lower high → short
# Works in both bull and bear markets by identifying reversals at extremes.
# Low trade frequency expected due to strict divergence requirements.

name = "12h_1d_momentum_divergence_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track recent extremes for divergence detection
    lookback = 10  # Look back 10 bars for swing points
    
    for i in range(lookback, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(ema_21_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Find recent swing low and high
        lookback_start = max(0, i - lookback)
        recent_low_idx = np.argmin(low[lookback_start:i+1]) + lookback_start
        recent_high_idx = np.argmax(high[lookback_start:i+1]) + lookback_start
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= lookback * 2:  # Need enough history
            # Current low vs previous low
            prev_low_start = max(0, i - lookback * 2)
            prev_low_end = i - lookback
            if prev_low_end > prev_low_start:
                prev_low_idx = np.argmin(low[prev_low_start:prev_low_end]) + prev_low_start
                if low[i] < low[prev_low_idx] and rsi[i] > rsi[prev_low_idx]:
                    bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= lookback * 2:
            # Current high vs previous high
            prev_high_start = max(0, i - lookback * 2)
            prev_high_end = i - lookback
            if prev_high_end > prev_high_start:
                prev_high_idx = np.argmax(high[prev_high_start:prev_high_end]) + prev_high_start
                if high[i] > high[prev_high_idx] and rsi[i] < rsi[prev_high_idx]:
                    bearish_div = True
        
        # Trend filter: align with higher timeframe trend
        bullish_signal = bullish_div and close[i] > ema_21_1d_aligned[i]
        bearish_signal = bearish_div and close[i] < ema_21_1d_aligned[i]
        
        # Exit on opposite divergence
        exit_long = bearish_div
        exit_short = bullish_div
        
        if bullish_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals