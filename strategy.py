#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Hypothesis: Weekly trend filters out false breakouts in opposing trends, volume confirms
# institutional participation, and Donchian breakouts capture momentum. Works in bull
# via upward breakouts, in bear via downward breakdowns, and avoids ranges via volume filter.
# Target: 10-20 trades/year to minimize fee drag.
name = "1d_donchian20_weekly_trend_volume_v1"
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
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily Donchian channels (20-period)
    # Highest high and lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: daily volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_ema_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high with volume and bullish weekly trend
            if close[i] > donchian_high[i] and vol_confirm and close[i] > weekly_ema_1d[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low with volume and bearish weekly trend
            elif close[i] < donchian_low[i] and vol_confirm and close[i] < weekly_ema_1d[i]:
                position = -1
                signals[i] = -0.25
    
    return signals