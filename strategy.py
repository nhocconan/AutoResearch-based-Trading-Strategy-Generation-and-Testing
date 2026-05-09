#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA20 trend filter and volume confirmation
# Combines long-term trend following with volatility breakout on daily timeframe.
# Works in both bull and bear markets by requiring trend alignment from 1w EMA20.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
# Uses volume confirmation to avoid false breakouts in low-volume environments.

name = "1d_Donchian20_1wEMA20_Trend_Volume"
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
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian(20) channels from daily data
    # Using 20-period high/low for breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above 20-day high
        short_breakout = close[i] < donchian_low[i-1]  # Break below 20-day low
        
        trend_up = close[i] > ema_20_1d[i]
        trend_down = close[i] < ema_20_1d[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if long_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif short_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below 20-day low or trend reversal
            if close[i] < donchian_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above 20-day high or trend reversal
            if close[i] > donchian_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals