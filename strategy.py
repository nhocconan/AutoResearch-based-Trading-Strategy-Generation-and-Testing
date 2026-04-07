#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Donchian(20) breakouts capture strong trends, filtered by weekly EMA trend
# and volume confirmation to avoid false breakouts. Works in both bull and bear markets
# by following the dominant trend on higher timeframes. Target: 12-30 trades/year.
name = "12h_donchian20_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    # Get weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_ema_12h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < weekly_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > weekly_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Volume confirmation required
            if volume_ok[i]:
                # Enter long: price breaks above Donchian high and bullish weekly trend
                if close[i] > donchian_high[i] and close[i] > weekly_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price breaks below Donchian low and bearish weekly trend
                elif close[i] < donchian_low[i] and close[i] < weekly_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals