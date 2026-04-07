#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout + Weekly Trend + Volume Confirmation
# Hypothesis: Donchian breakouts capture strong directional moves. Weekly trend filter ensures
# alignment with higher timeframe momentum. Volume confirmation filters false breakouts.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Target: 20-30 trades/year to minimize fee drag on daily timeframe.
name = "1d_donchian_breakout_weekly_trend_volume_v2"
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
    
    # Donchian Channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter (more responsive)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_ema_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian channel or weekly trend turns bearish
            if close[i] < lowest_low[i] or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian channel or weekly trend turns bullish
            if close[i] > highest_high[i] or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Enter long: price breaks above upper Donchian, volume confirms, bullish weekly trend
            if (close[i] > highest_high[i] and volume_confirm and 
                close[i] > weekly_ema_1d[i]):
                position = 1
                signals[i] = 0.30
            # Enter short: price breaks below lower Donchian, volume confirms, bearish weekly trend
            elif (close[i] < lowest_low[i] and volume_confirm and 
                  close[i] < weekly_ema_1d[i]):
                position = -1
                signals[i] = -0.30
    
    return signals