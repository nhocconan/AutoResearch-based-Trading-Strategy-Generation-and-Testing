#!/usr/bin/env python3
# 12h_RSI2_Overbought_Oversold
# Hypothesis: Mean reversion on 12h timeframe using extreme RSI levels (2-period RSI).
# Long when RSI(2) < 10 and price above 200-period EMA (uptrend filter).
# Short when RSI(2) > 90 and price below 200-period EMA (downtrend filter).
# Uses 1d trend filter: only trade in direction of daily EMA200 trend.
# Works in bull/bear by following higher timeframe trend and using RSI extremes for mean reversion.
# Target: 15-30 trades/year per symbol.

name = "12h_RSI2_Overbought_Oversold"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 2-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi2 = 100 - (100 / (1 + rs))
    
    # 200-period EMA for trend filter
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_uptrend = close_1d > ema200_1d
    daily_downtrend = close_1d < ema200_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi2[i]) or np.isnan(ema200[i]) or 
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi2[i]
        price_above_ema = close[i] > ema200[i]
        price_below_ema = close[i] < ema200[i]
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + RSI(2) oversold + price above EMA200
            if daily_up and rsi_val < 10 and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + RSI(2) overbought + price below EMA200
            elif daily_down and rsi_val > 90 and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or trend changes
            if rsi_val > 50 or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or trend changes
            if rsi_val < 50 or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals