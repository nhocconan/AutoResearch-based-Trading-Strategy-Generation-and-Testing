#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout + Weekly Trend + Volume Confirmation
# Hypothesis: Daily Donchian(20) breakouts with weekly trend filter and volume
# confirmation capture institutional breakouts. Works in bull (breakouts up) and
# bear (breakdowns down). Weekly trend prevents counter-trend trades. Volume
# ensures legitimacy. Target: 15-25 trades/year to minimize fee drag.
name = "daily_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous day's data
    # Use 20-period high/low from previous days
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # 20-period rolling high/low of previous day's high/low
    # This gives us the Donchian channel based on prior 20 days
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_high_d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_d = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_ema_d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: daily volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_d[i]) or np.isnan(donchian_low_d[i]) or
            np.isnan(weekly_ema_d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low_d[i] or close[i] < weekly_ema_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high_d[i] or close[i] > weekly_ema_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high (breakout) with volume and bullish weekly trend
            if close[i] > donchian_high_d[i] and vol_confirm and close[i] > weekly_ema_d[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low (breakdown) with volume and bearish weekly trend
            elif close[i] < donchian_low_d[i] and vol_confirm and close[i] < weekly_ema_d[i]:
                position = -1
                signals[i] = -0.25
    
    return signals