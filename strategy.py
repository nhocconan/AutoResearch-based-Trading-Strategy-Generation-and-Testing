#!/usr/bin/env python3
# 6h_RSI_Divergence_1dTrendFilter
# Hypothesis: On 6h timeframe, enter long when RSI(14) makes a bullish divergence (higher low in RSI vs lower low in price) and price is above 1d EMA50.
# Enter short when RSI makes a bearish divergence (lower high in RSI vs higher high in price) and price is below 1d EMA50.
# Exit when price crosses 1d EMA50 (trend reversal).
# Uses divergence to catch reversals in both bull and bear markets, with EMA filter to avoid counter-trend trades.
# Targets 15-25 trades/year for low fee drag.

name = "6h_RSI_Divergence_1dTrendFilter"
timeframe = "6h"
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
    
    # Load daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Find local minima and maxima for divergence detection
    # We'll look for divergences over the last 5 bars
    rsi_min = np.full(n, np.nan)
    rsi_max = np.full(n, np.nan)
    price_min = np.full(n, np.nan)
    price_max = np.full(n, np.nan)
    
    lookback = 5
    
    for i in range(lookback, n):
        # Find RSI and price minima in lookback window
        rsi_window = rsi[i-lookback:i+1]
        price_window_low = low[i-lookback:i+1]
        price_window_high = high[i-lookback:i+1]
        
        min_idx = np.argmin(rsi_window)
        max_idx = np.argmax(rsi_window)
        
        rsi_min[i] = rsi_window[min_idx]
        rsi_max[i] = rsi_window[max_idx]
        price_min[i] = price_window_low[min_idx]
        price_max[i] = price_window_high[max_idx]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(rsi_min[i]) or np.isnan(rsi_max[i]) or
            np.isnan(price_min[i]) or np.isnan(price_max[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema1d_trend = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        rsi_min_val = rsi_min[i]
        rsi_max_val = rsi_max[i]
        price_min_val = price_min[i]
        price_max_val = price_max[i]
        
        if position == 0:
            # Bullish divergence: RSI makes higher low while price makes lower low
            bull_div = (rsi_min_val > rsi[i-lookback]) and (price_min_val < low[i-lookback])
            # Bearish divergence: RSI makes lower high while price makes higher high
            bear_div = (rsi_max_val < rsi[i-lookback]) and (price_max_val > high[i-lookback])
            
            if bull_div and close[i] > ema1d_trend:
                signals[i] = 0.25
                position = 1
            elif bear_div and close[i] < ema1d_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA50 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA50 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals