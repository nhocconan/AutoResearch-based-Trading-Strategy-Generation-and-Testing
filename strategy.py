#!/usr/bin/env python3
# Hypothesis: Daily price crosses above/below 1-week VWAP with volume confirmation and trend filter.
# Uses weekly VWAP as dynamic support/resistance: long when price > VWAP with increasing volume,
# short when price < VWAP with decreasing volume. Trend filter uses daily EMA50 to avoid counter-trend trades.
# Target: 20-60 total trades over 4 years (5-15/year) with size 0.25.

name = "1d_VWAP_Trend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for each weekly bar
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_values = vwap_1w.values
    
    # Align VWAP to daily timeframe (waits for weekly close)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_values)
    
    # Daily EMA50 for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume change: current vs previous day
    volume_change = volume - np.roll(volume, 1)
    volume_change[0] = 0  # First value has no previous
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if VWAP not available (first weekly bar)
        if np.isnan(vwap_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > weekly VWAP AND volume increasing AND price > EMA50 (uptrend)
            if close[i] > vwap_1w_aligned[i] and volume_change[i] > 0 and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price < weekly VWAP AND volume decreasing AND price < EMA50 (downtrend)
            elif close[i] < vwap_1w_aligned[i] and volume_change[i] < 0 and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP OR trend turns down (price < EMA50)
            if close[i] < vwap_1w_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP OR trend turns up (price > EMA50)
            if close[i] > vwap_1w_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals