#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Breakout with 12h EMA Trend and Volume Confirmation
# Goes long when price breaks above upper Bollinger Band (20,2) with price > 12h EMA50 and volume > 1.5x average
# Goes short when price breaks below lower Bollinger Band with price < 12h EMA50 and volume > 1.5x average
# Exits when price returns to middle Bollinger Band or trend reverses
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets with proper trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # EMA50 on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after Bollinger Band warmup
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(basis[i]) or
            np.isnan(ema_12h_aligned[i])):
            continue
        
        # Long entry: price breaks above upper BB + price > 12h EMA50 + volume confirmation
        if (close[i] > upper[i] and
            close[i] > ema_12h_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower BB + price < 12h EMA50 + volume confirmation
        elif (close[i] < lower[i] and
              close[i] < ema_12h_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to middle Bollinger Band or trend reverses
        elif position == 1 and (close[i] < basis[i] or close[i] < ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > basis[i] or close[i] > ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0