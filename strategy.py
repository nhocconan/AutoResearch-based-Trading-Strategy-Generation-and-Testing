#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Donchian breakout direction and 1d trend filter, plus volume confirmation.
# Uses 4h for signal direction (Donchian breakout), 1d for trend filter (EMA50), 1h for entry timing and volume.
# Designed for low trade frequency (15-37/year) to avoid fee drag. Works in bull/bear by requiring alignment
# with higher timeframe trend and volume spikes to confirm institutional interest.
name = "1h_Donchian4h_1dEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_4h = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    donchian_low_4h = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        bullish_breakout = close[i] > donchian_high_4h[i-1]  # Break above 4h Donchian high
        bearish_breakout = close[i] < donchian_low_4h[i-1]   # Break below 4h Donchian low
        trend_up = close[i] > ema_50_1h[i]
        trend_down = close[i] < ema_50_1h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if bullish_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif bearish_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or trend reversal
            if bearish_breakout or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: bullish breakout or trend reversal
            if bullish_breakout or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals