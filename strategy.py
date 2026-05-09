#!/usr/bin/env python3
# Hypothesis: 1-day timeframe with 1-week RSI trend filter and weekly Donchian breakout confirmation.
# Enters long when daily RSI < 40 (oversold) and price > weekly Donchian upper (breakout).
# Enters short when daily RSI > 60 (overbought) and price < weekly Donchian lower (breakdown).
# Uses 1-week RSI (14) as trend filter: only allow longs when weekly RSI > 50, shorts when weekly RSI < 50.
# Exits when RSI reverts to neutral (40-60) or breakout fails.
# Target: 20-60 total trades over 4 years (5-15/year) with size 0.25.

name = "1D_RSI_Oversold_WeeklyDonchian_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1-week RSI (14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    rsi_1w_50 = 50.0  # threshold for trend filter
    
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate daily RSI (14) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly Donchian channel (20-period) for breakout confirmation
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    donchian_upper = high_1w.rolling(window=20, min_periods=20).max()
    donchian_lower = low_1w.rolling(window=20, min_periods=20).min()
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: daily RSI < 40 (oversold) + weekly RSI > 50 (uptrend) + price > weekly Donchian upper
            if rsi[i] < 40 and rsi_1w_aligned[i] > rsi_1w_50 and close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: daily RSI > 60 (overbought) + weekly RSI < 50 (downtrend) + price < weekly Donchian lower
            elif rsi[i] > 60 and rsi_1w_aligned[i] < rsi_1w_50 and close[i] < donchian_lower_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: daily RSI > 50 (revert to neutral) OR price < weekly Donchian lower (breakdown)
            if rsi[i] > 50 or close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: daily RSI < 50 (revert to neutral) OR price > weekly Donchian upper (breakout)
            if rsi[i] < 50 or close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals