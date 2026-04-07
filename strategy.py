#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian 20 breakout with volume filter
# Hypothesis: Donchian(20) breakouts on 12h timeframe capture strong momentum moves.
# Volume filter ensures only institutional participation triggers entries.
# Works in both bull and bear markets: In bull, breakouts above upper band continue up;
# in bear, breakouts below lower band continue down. Uses 1d timeframe for trend filter.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_donchian20_volume_trend_v1"
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
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_daily['close'].values
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower or trend turns bearish
            if close[i] < donchian_lower[i] or close[i] < ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper or trend turns bullish
            if close[i] > donchian_upper[i] or close[i] > ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper with volume and bullish trend
            if high[i] > donchian_upper[i] and close[i] > donchian_upper[i] and vol_filter[i] and close[i] > ema50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower with volume and bearish trend
            elif low[i] < donchian_lower[i] and close[i] < donchian_lower[i] and vol_filter[i] and close[i] < ema50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals