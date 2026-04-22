#!/usr/bin/env python3
"""
Hypothesis: 4-hour RSI(2) mean reversion with 1-day volatility filter and volume confirmation.
Long when RSI(2) < 10, price > 1d EMA200 (uptrend filter), and volume > 1.5x 20-period average.
Short when RSI(2) > 90, price < 1d EMA200 (downtrend filter), and volume > 1.5x 20-period average.
Exit when RSI(2) crosses above 50 (long) or below 50 (short).
Uses extreme RSI(2) for mean reversion in both bull and bear markets, with trend filter to avoid counter-trend trades.
Designed for low trade frequency (<50/year) to minimize fee drag.
"""

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
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    daily_close = df_1d['close'].values
    daily_ema200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    daily_ema200_aligned = align_htf_to_ltf(prices, df_1d, daily_ema200)
    
    # Calculate RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(daily_ema200_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) + price > daily EMA200 (uptrend) + volume filter
            if (rsi[i] < 10 and 
                close[i] > daily_ema200_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) + price < daily EMA200 (downtrend) + volume filter
            elif (rsi[i] > 90 and 
                  close[i] < daily_ema200_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI(2) crosses above 50
                if rsi[i] > 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI(2) crosses below 50
                if rsi[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_RSI2_MeanReversion_VolumeTrendFilter"
timeframe = "4h"
leverage = 1.0