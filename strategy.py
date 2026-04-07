#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout + Daily Trend + Volume Confirmation
# Hypothesis: Breakouts above 20-period highs/lows with daily trend filter and volume
# confirmation capture institutional moves. Works in bull via upward breakouts, in bear via
# downward breakdowns, and avoids false breakouts via trend/volume filters.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_donchian20_daily_trend_volume_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(21) for trend
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=21, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 12h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or daily trend turns bearish
            if close[i] < donchian_low[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or daily trend turns bullish
            if close[i] > donchian_high[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high with volume and bullish daily trend
            if close[i] > donchian_high[i] and vol_confirm and close[i] > daily_ema_12h[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low with volume and bearish daily trend
            elif close[i] < donchian_low[i] and vol_confirm and close[i] < daily_ema_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals