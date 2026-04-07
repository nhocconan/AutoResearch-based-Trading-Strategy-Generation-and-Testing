#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout with Volume Filter and Daily Trend Filter
# Hypothesis: On 12h timeframe, Donchian(20) breakouts indicate strong momentum.
# Combines with daily trend filter (price vs 50 EMA) to align with higher timeframe direction.
# Volume filter ensures breakouts have institutional participation.
# Works in both bull and bear markets by only taking breakouts in direction of daily trend.
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
    
    # Daily trend: 50 EMA on close
    close_series_daily = pd.Series(df_daily['close'].values)
    ema_50_daily = close_series_daily.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Donchian(20) on 12h high/low
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_daily_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or trend turns bearish or volume drops
            if (close[i] < donchian_low[i] or close[i] < ema_50_daily_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend turns bullish or volume drops
            if (close[i] > donchian_high[i] or close[i] > ema_50_daily_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and bullish daily trend
            if ((high[i] > donchian_high[i] or close[i] > donchian_high[i]) and 
                close[i] > ema_50_daily_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and bearish daily trend
            elif ((low[i] < donchian_low[i] or close[i] < donchian_low[i]) and 
                  close[i] < ema_50_daily_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals