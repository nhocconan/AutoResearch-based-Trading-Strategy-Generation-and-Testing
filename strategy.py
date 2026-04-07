#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Captures momentum in trending markets while avoiding whipsaws.
# Weekly trend filter ensures alignment with higher timeframe direction.
# Volume confirmation filters for institutional participation.
# Designed for low frequency: target 7-25 trades/year (30-100 total over 4 years).
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

name = "1d_donchian20_1w_trend_volume_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_1w = close_1w.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
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
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period low
        
        # Weekly trend filter: price above/below EMA(50)
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
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