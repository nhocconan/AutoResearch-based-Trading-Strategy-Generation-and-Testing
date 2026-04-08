#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets while avoiding whipsaws.
# Daily trend filter ensures trades align with higher timeframe direction.
# Volume confirmation filters for institutional participation.
# Designed for low frequency: target 12-37 trades/year to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

name = "12h_donchian20_1d_trend_volume_v1"
timezone = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Daily trend filter: price above/below EMA(50)
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: opposite Donchian break
        exit_long = close[i] < donchian_low[i-1]
        exit_short = close[i] > donchian_high[i-1]
        
        if position == 1:  # Long position
            # Exit on breakdown or trend reversal
            if exit_long or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on breakout or trend reversal
            if exit_short or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: upward breakout + uptrend + volume confirmation
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: downward breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals