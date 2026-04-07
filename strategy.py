#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian20 with Daily Trend Filter and Volume Confirmation
# Hypothesis: Donchian channel breakouts capture trend continuation. Using daily EMA200 as trend filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws in choppy markets. Volume confirmation ensures breakouts have institutional participation. Works in both bull and bear markets: In bull, we take long breakouts above upper band; in bear, we take short breakouts below lower band. The trend filter prevents counter-trend trades during strong reversals.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_donchian20_daily_trend_volume_v1"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_daily = df_daily['close'].values
    ema200_daily = pd.Series(close_daily).ewm(span=200, adjust=False).mean().values
    ema200_daily_aligned = align_htf_to_ltf(prices, df_daily, ema200_daily)
    
    # Calculate Donchian channels (20-period) on 12h data
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema200_daily_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or trend changes or volume drops
            if close[i] <= donchian_low[i] or close[i] < ema200_daily_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend changes or volume drops
            if close[i] >= donchian_high[i] or close[i] > ema200_daily_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and uptrend
            if high[i] > donchian_high[i] and close[i] > donchian_high[i] and vol_filter[i] and close[i] > ema200_daily_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and downtrend
            elif low[i] < donchian_low[i] and close[i] < donchian_low[i] and vol_filter[i] and close[i] < ema200_daily_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals