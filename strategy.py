#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume and Trend Filter
# Hypothesis: Price breaking Donchian(20) channels on 4h with volume confirmation and 
# trend filter (price vs 200 EMA) works in both bull and bear markets.
# In bull markets: buy breakouts above upper band, sell breakdowns below lower band.
# In bear markets: sell breakdowns below lower band, buy breakouts above upper band.
# Uses 1d timeframe for trend filter only to avoid look-ahead.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian20_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate 200 EMA on daily close (using close prices)
    daily_close = df_daily['close'].values
    daily_close_series = pd.Series(daily_close)
    ema_200_daily = daily_close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align daily EMA to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_daily, ema_200_daily)
    
    # Calculate Donchian channels on 4h data (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel or trend fails
            if close[i] <= donchian_high[i] or close[i] < ema_200_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel or trend fails
            if close[i] >= donchian_low[i] or close[i] > ema_200_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above upper Donchian band with volume and trend
            if high[i] > donchian_high[i] and close[i] > donchian_high[i] and \
               close[i] > ema_200_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower Donchian band with volume and trend
            elif low[i] < donchian_low[i] and close[i] < donchian_low[i] and \
                 close[i] < ema_200_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals