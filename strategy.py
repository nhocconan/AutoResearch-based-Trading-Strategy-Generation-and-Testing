#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian breakout + weekly trend + volume confirmation
# Hypothesis: Donchian(20) breakouts on 12h capture strong directional moves.
# Weekly EMA(21) filter ensures we only trade with the long-term trend.
# Volume confirmation (volume > 1.5x average) filters false breakouts.
# Works in both bull and bear markets by following the weekly trend direction.
# Target: 15-25 trades/year to minimize fee drag on 12h timeframe.
name = "12h_donchian20_weekly_trend_volume_v1"
timeframe = "12h"
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily trend filter (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(21) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=21, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(daily_ema_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower band or daily trend turns bearish
            if close[i] < lowest_low[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper band or daily trend turns bullish
            if close[i] > highest_high[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Volume confirmation: volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Enter long: price breaks above Donchian upper band + volume + bullish daily trend
            if (close[i] > highest_high[i]) and vol_confirm and (close[i] > daily_ema_12h[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band + volume + bearish daily trend
            elif (close[i] < lowest_low[i]) and vol_confirm and (close[i] < daily_ema_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals