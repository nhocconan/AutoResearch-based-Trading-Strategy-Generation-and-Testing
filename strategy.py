#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian breakout with weekly trend filter and volume confirmation
# Hypothesis: Donchian breakouts capture institutional moves; weekly trend filter ensures
# alignment with higher timeframe momentum; volume confirms participation.
# Works in bull via upward breakouts, in bear via downward breakdowns.
# Target: 10-25 trades/year to minimize fee drag on daily timeframe.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily 20-period Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily 20-period volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (breakdown)
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (breakout)
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian high + volume + weekly uptrend
            if (close[i] > donch_high[i] and vol_confirm and above_weekly_ema):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low + volume + weekly downtrend
            elif (close[i] < donch_low[i] and vol_confirm and below_weekly_ema):
                position = -1
                signals[i] = -0.25
    
    return signals