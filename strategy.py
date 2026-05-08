#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h EMA21 trend filter and 4h RSI14 mean reversion
# Uses 4h EMA21 for trend direction and 4h RSI14 for mean reversion entries.
# Long when price > EMA21 and RSI < 30; short when price < EMA21 and RSI > 70.
# Session filter (08-20 UTC) to avoid low-volume periods. Position size 0.20.
# Designed to work in both bull and bear markets by following 4h trend while
# entering on overextended moves. Target: 60-150 total trades over 4 years.

name = "1h_EMA21_RSI14_MeanReversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA and RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA21
    close_4h = df_4h['close'].values
    ema21_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 21:
        ema21_4h[20] = np.mean(close_4h[:21])
        for i in range(21, len(close_4h)):
            ema21_4h[i] = (close_4h[i] * 2 + ema21_4h[i-1] * 19) / 21
    
    # Calculate 4h RSI14
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_4h), np.nan)
    avg_loss = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_4h)):
            avg_gain[i] = (gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(len(close_4h))
    rsi_4h = np.full(len(close_4h), np.nan)
    for i in range(14, len(close_4h)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_4h[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_4h[i] = 100 if avg_gain[i] > 0 else 0
    
    # Align 4h indicators to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry: follow 4h EMA trend with RSI mean reversion
            # Long when price above EMA21 and RSI oversold (<30)
            long_condition = (
                close[i] > ema21_4h_aligned[i] and   # price above 4h EMA21 (bullish bias)
                rsi_4h_aligned[i] < 30               # RSI oversold
            )
            
            # Short when price below EMA21 and RSI overbought (>70)
            short_condition = (
                close[i] < ema21_4h_aligned[i] and   # price below 4h EMA21 (bearish bias)
                rsi_4h_aligned[i] > 70               # RSI overbought
            )
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA21 or RSI becomes overbought
            if close[i] < ema21_4h_aligned[i] or rsi_4h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above EMA21 or RSI becomes oversold
            if close[i] > ema21_4h_aligned[i] or rsi_4h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals