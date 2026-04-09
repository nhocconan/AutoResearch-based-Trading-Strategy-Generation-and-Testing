#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# Uses daily Donchian channels for structure, breaks above/below 20-period high/low for entries
# Only takes breakouts when 1w EMA(21) is above/below price for trend alignment
# Requires volume > 1.5x 20-day average for confirmation
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in both bull/bear: 1w EMA filter ensures we trade with the major trend, reducing false breakouts in ranging markets

name = "1d_1w_donchian_volume_ema_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low) using previous period to avoid look-ahead
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(df_1d['high'].iloc[i-20:i])
            donchian_low[i] = np.min(df_1d['low'].iloc[i-20:i])
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        multiplier = 2 / (21 + 1)
        ema_1w[20] = np.mean(close_1w[0:21])  # SMA seed
        for i in range(21, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Donchian channels to 1d timeframe (already aligned, but ensure proper shifting)
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_1d[i]) or 
            np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low OR trend turns bearish (price < EMA)
            if close[i] < donchian_low_1d[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high OR trend turns bullish (price > EMA)
            if close[i] > donchian_high_1d[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and trend filter
            if volume_confirm:
                # Long breakout: price closes above Donchian high AND price > EMA (bullish trend)
                if close[i] > donchian_high_1d[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low AND price < EMA (bearish trend)
                elif close[i] < donchian_low_1d[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals