#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-period RSI with 4h/1d trend filter and session filter (08-20 UTC)
# Long when: RSI(4) < 30 AND close > 4h EMA50 AND close > 1d EMA200 AND in session
# Short when: RSI(4) > 70 AND close < 4h EMA50 AND close < 1d EMA200 AND in session
# Exit when: RSI(4) crosses back above 50 (for long) or below 50 (for short)
# Uses RSI extremes for mean reversion in higher timeframe trends, session filter to avoid low-liquidity hours
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_RSI4_4hEMA50_1dEMA200_Session"
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
    open_time = prices['open_time'].values
    
    # Get 4h data ONCE before loop for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # need sufficient data for EMA200
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1h RSI(4)
    if len(close) >= 5:  # need at least 5 bars for RSI(4)
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
        avg_loss = loss.ewm(alpha=1/4, adjust=False, min_periods=4).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_values = rsi.values
    else:
        rsi_values = np.full(n, 50.0)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI(4) < 30 AND above 4h EMA50 AND above 1d EMA200 AND in session
            if (rsi_values[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI(4) > 70 AND below 4h EMA50 AND below 1d EMA200 AND in session
            elif (rsi_values[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI(4) crosses back above 50
            if rsi_values[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI(4) crosses back below 50
            if rsi_values[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals