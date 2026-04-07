#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly KAMA with Volume and Trend Filter
# Hypothesis: Price reacting to weekly KAMA direction with volume confirmation and trend filter (price vs 200 EMA)
# In bull markets: buy when price > KAMA and rising, sell when price < KAMA and falling
# In bear markets: sell when price < KAMA and falling, buy when price > KAMA and rising
# Target: 20-50 trades/year (80-200 over 4 years)

name = "1d_weekly_kama_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly KAMA (Kaufman Adaptive Moving Average)
    weekly_close = df_weekly['close'].values
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(weekly_close, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(weekly_close, 1)), axis=0)  # 10-period volatility
    # Handle first 10 elements
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/3 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(weekly_close, np.nan)
    kama[9] = weekly_close[9]  # Start at index 9
    for i in range(10, len(weekly_close)):
        if np.isnan(kama[i-1]):
            kama[i] = weekly_close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    kama = np.roll(kama, 1)
    if len(kama) > 1:
        kama[0] = kama[1]
    else:
        kama[0] = 0
    
    # Align to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price crosses below KAMA or trend fails
            if close[i] < kama_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: price crosses above KAMA or trend fails
            if close[i] > kama_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price crosses above KAMA with volume and trend
            if close[i] > kama_aligned[i] and close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below KAMA with volume and trend
            elif close[i] < kama_aligned[i] and close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals