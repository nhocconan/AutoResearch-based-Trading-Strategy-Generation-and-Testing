#!/usr/bin/env python3
# 4h_RSI_Divergence_Trend_Pullback
# Hypothesis: Combine 4-hour RSI divergence with 1-day EMA trend filter and volume confirmation.
# RSI divergence identifies potential reversals with momentum exhaustion.
# The 1d EMA200 filter ensures trades align with long-term trend, reducing false signals.
# Volume confirmation adds conviction to reversal signals.
# Works in bull markets (buy bullish divergence in uptrend) and bear markets (sell bearish divergence in downtrend).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4h_RSI_Divergence_Trend_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate RSI divergence signals
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 14
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence
        if (low[i] < low[i-lookback] and 
            rsi[i] > rsi[i-lookback]):
            # Check if it's a meaningful divergence
            price_lower = low[i] == np.min(low[i-lookback:i+1])
            rsi_higher = rsi[i] == np.max(rsi[i-lookback:i+1])
            if price_lower and rsi_higher:
                bullish_div[i] = True
        
        # Bearish divergence
        if (high[i] > high[i-lookback] and 
            rsi[i] < rsi[i-lookback]):
            # Check if it's a meaningful divergence
            price_higher = high[i] == np.max(high[i-lookback:i+1])
            rsi_lower = rsi[i] == np.min(rsi[i-lookback:i+1])
            if price_higher and rsi_lower:
                bearish_div[i] = True
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bullish RSI divergence + price above 1d EMA200 (bullish trend) + volume spike
            if (bullish_div[i] and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: bearish RSI divergence + price below 1d EMA200 (bearish trend) + volume spike
            elif (bearish_div[i] and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish RSI divergence or price below 1d EMA200
            if (bearish_div[i] or close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish RSI divergence or price above 1d EMA200
            if (bullish_div[i] or close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals