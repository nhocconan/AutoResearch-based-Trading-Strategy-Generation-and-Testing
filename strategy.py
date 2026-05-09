#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Works in both bull and bear markets by requiring alignment with daily trend and high-volume breakouts.
# Uses proven Donchian breakout structure for trend continuation with low trade frequency.
name = "4h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from previous 1d bar (20-period high/low)
    donchian_high = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donchian_low = df_1d['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 2.0x 20-period average volume (strict to reduce trades)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous day's data for Donchian levels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        bullish_breakout = close[i] > donchian_high_4h[i]  # Break above upper band
        bearish_breakout = close[i] < donchian_low_4h[i]   # Break below lower band
        trend_up = close[i] > ema_50_4h[i]
        trend_down = close[i] < ema_50_4h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if bullish_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif bearish_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or trend reversal
            if bearish_breakout or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout or trend reversal
            if bullish_breakout or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals