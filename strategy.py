#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 12h Trend Filter and Volume Confirmation
# Hypothesis: Price breaking above/below Donchian(20) channels with strong volume
# and aligned 12h trend captures institutional moves in both bull/bear markets.
# 12h EMA(50) determines trend direction; only trade breakouts in trend direction.
# Volume > 1.5x 20-period average confirms institutional participation.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period) on 4h data
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
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if (close[i] <= donchian_low[i] or close[i] < ema_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if (close[i] >= donchian_high[i] or close[i] > ema_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and uptrend
            if ((high[i] > donchian_high[i] or close[i] > donchian_high[i]) and 
                vol_filter[i] and close[i] > ema_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and downtrend
            elif ((low[i] < donchian_low[i] or close[i] < donchian_low[i]) and 
                  vol_filter[i] and close[i] < ema_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals