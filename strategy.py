#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and Trend Filter
# Hypothesis: Price breaking above/below 4-hour Donchian channels indicates momentum.
# Volume confirms institutional participation. EMA trend filter ensures alignment with higher timeframe.
# Works in both bull and bear markets: only take long signals in bull (price > 200 EMA),
# only short signals in bear (price < 200 EMA). This reduces whipsaw.
# Target: 25-50 trades/year (100-200 over 4 years).

name = "4h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: price above/below 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or trend turns bearish or volume drops
            if (low[i] < donchian_low[i] or close[i] < ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend turns bullish or volume drops
            if (high[i] > donchian_high[i] or close[i] > ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and bullish trend (price > 200 EMA)
            if (high[i] > donchian_high[i] and close[i] > ema_200[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and bearish trend (price < 200 EMA)
            elif (low[i] < donchian_low[i] and close[i] < ema_200[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals