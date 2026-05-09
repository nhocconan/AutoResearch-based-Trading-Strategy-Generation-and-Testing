#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Combines Donchian channel breakout (price channel) with 1d EMA50 trend filter and volume confirmation
# Designed for 12h timeframe with target of 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear markets by requiring trend alignment and volume confirmation
name = "12h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian(20) channels
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA and Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h[i]) or np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_12h[i-1]  # Break above upper band
        short_breakout = close[i] < donchian_low_12h[i-1]  # Break below lower band
        
        trend_up = close[i] > ema_50_12h[i]
        trend_down = close[i] < ema_50_12h[i]
        
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
            # Exit long: bearish breakout below lower band or trend reversal
            if close[i] < donchian_low_12h[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above upper band or trend reversal
            if close[i] > donchian_high_12h[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals