#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA20 trend filter and volume confirmation
# Combines breakout logic from daily Donchian channels with trend filter from weekly EMA20
# and volume confirmation to reduce false breakouts. Designed for 1d timeframe with
# target of 30-100 trades over 4 years (7-25/year). Works in bull/bear markets by
# requiring trend alignment and volume confirmation.
name = "1d_Donchian20_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Get 1d data for Donchian(20) channels (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d (already aligned, but keep for consistency)
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d[i]) or np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_1d[i-1]  # Break above upper band
        short_breakout = close[i] < donchian_low_1d[i-1]  # Break below lower band
        
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
            # Exit long: bearish breakout below lower band or trend reversal
            if close[i] < donchian_low_1d[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above upper band or trend reversal
            if close[i] > donchian_high_1d[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals