#!/usr/bin/env python3
# 12H_1D_Adaptive_RSI_Divergence
# Hypothesis: On 12h timeframe, RSI divergence from price at extreme levels (oversold/overbought)
# combined with 1d trend filter captures reversals in both bull and bear markets.
# Uses 1d EMA50 for trend direction and 12h RSI(14) with bullish/bearish divergence detection.
# Divergence occurs when price makes new high/low but RSI does not, indicating weakening momentum.
# Works in trending markets by catching pullbacks and in ranging markets by catching reversals.
# Target: 20-30 trades/year per symbol with low turnover to minimize fee drag.

name = "12H_1D_Adaptive_RSI_Divergence"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align trend to 12h
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate RSI on 12h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Initialize before smoothing period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Lookback period for divergence detection
    lookback = 10
    
    for i in range(lookback, n):
        # Skip if trend data not ready
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Check for bullish divergence: price makes lower low, RSI makes higher low
            if bullish:
                # Find lowest price in lookback window
                lookback_low_idx = np.argmin(low[i-lookback:i+1])
                abs_low_idx = i - lookback + lookback_low_idx
                
                # Current price is new low compared to lookback period
                if low[i] <= low[abs_low_idx] and abs_low_idx < i:
                    # Check if RSI at current point is higher than RSI at lookback low
                    if rsi[i] > rsi[abs_low_idx] and rsi[i] < 40:  # Oversold condition
                        signals[i] = 0.25
                        position = 1
            
            # Check for bearish divergence: price makes higher high, RSI makes lower high
            elif bearish:
                # Find highest price in lookback window
                lookback_high_idx = np.argmax(high[i-lookback:i+1])
                abs_high_idx = i - lookback + lookback_high_idx
                
                # Current price is new high compared to lookback period
                if high[i] >= high[abs_high_idx] and abs_high_idx < i:
                    # Check if RSI at current point is lower than RSI at lookback high
                    if rsi[i] < rsi[abs_high_idx] and rsi[i] > 60:  # Overbought condition
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit long: bearish trend or RSI becomes overbought
            if bearish or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or RSI becomes oversold
            if bullish or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals