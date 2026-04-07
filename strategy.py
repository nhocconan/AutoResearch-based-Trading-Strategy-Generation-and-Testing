#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Breakouts from weekly Donchian channels capture major trend moves,
# daily volume confirms institutional participation, and weekly trend filter avoids counter-trend trades.
# Works in bull via upward breakouts above weekly channel, in bear via downward breakdowns below weekly channel.
# Target: 15-25 trades/year to minimize fee drag on daily timeframe.
name = "daily_donchian_weekly_trend_volume_v1"
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
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly 20-period Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate weekly 20-period EMA for trend filter
    ema_20 = pd.Series(high_1w).ewm(span=20, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Calculate daily 20-period volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above weekly Donchian high + above weekly EMA + volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_20_aligned[i] and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low + below weekly EMA + volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_20_aligned[i] and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals