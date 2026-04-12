#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_fib_retracement_v1
# Uses weekly Fibonacci retracement levels (38.2%, 61.8%) as support/resistance in trending markets.
# In bull markets: buy near 61.8% retracement of weekly uptrend; in bear markets: sell near 38.2% retracement of weekly downtrend.
# Filters: price must be above/below 200 EMA on daily for trend alignment, and volume must be above average.
# Low trade frequency expected (10-25/year) due to specific price levels and trend filter.
# Works in both bull and bear markets by trading pullbacks in the direction of the weekly trend.
name = "1d_1w_fib_retracement_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Fibonacci levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly high and low over last 52 weeks (approx 1 year)
    lookback = min(52, len(df_1w))
    weekly_high = np.max(df_1w['high'][-lookback:].values)
    weekly_low = np.min(df_1w['low'][-lookback:].values)
    weekly_range = weekly_high - weekly_low
    
    # Calculate Fibonacci levels
    fib_382 = weekly_low + 0.382 * weekly_range
    fib_618 = weekly_low + 0.618 * weekly_range
    
    # Align weekly Fib levels to daily
    fib_382_arr = np.full_like(df_1w['high'], fib_382)
    fib_618_arr = np.full_like(df_1w['high'], fib_618)
    fib_382_aligned = align_htf_to_ltf(prices, df_1w, fib_382_arr)
    fib_618_aligned = align_htf_to_ltf(prices, df_1w, fib_618_arr)
    
    # Daily 200 EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA warmup
        # Skip if any data not ready
        if np.isnan(ema_200[i]) or np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bullish setup: price near 61.8% fib level, above 200 EMA, volume above average
        near_fib_618 = abs(close[i] - fib_618_aligned[i]) / close[i] < 0.015  # Within 1.5%
        uptrend = close[i] > ema_200[i]
        vol_filter = volume[i] > vol_ma[i]
        
        # Bearish setup: price near 38.2% fib level, below 200 EMA, volume above average
        near_fib_382 = abs(close[i] - fib_382_aligned[i]) / close[i] < 0.015  # Within 1.5%
        downtrend = close[i] < ema_200[i]
        
        if near_fib_618 and uptrend and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif near_fib_382 and downtrend and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (close[i] < fib_382_aligned[i] or close[i] < ema_200[i]):
            # Exit long if price breaks below 38.2% or below EMA200
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > fib_618_aligned[i] or close[i] > ema_200[i]):
            # Exit short if price breaks above 61.8% or above EMA200
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