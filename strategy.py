#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and 1d Trend Filter v1
# Hypothesis: In both bull and bear markets, price tends to breakout from Donchian channels with momentum.
# We buy when price breaks above 20-period Donchian high with volume confirmation and daily trend alignment.
# We sell when price breaks below 20-period Donchian low with volume confirmation and daily trend alignment.
# Daily trend filter ensures we trade in the direction of the higher timeframe trend.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian_breakout_volume_1d_trend_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend direction
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    
    # Align daily EMA50 to 4h timeframe
    daily_ema50_4h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(daily_ema50_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (stop) or reaches Donchian high (take profit)
            if close[i] <= donchian_low[i] or close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (stop) or reaches Donchian low (take profit)
            if close[i] >= donchian_high[i] or close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price closes above Donchian high with daily uptrend
                if close[i] > donchian_high[i] and close[i] > daily_ema50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below Donchian low with daily downtrend
                elif close[i] < donchian_low[i] and close[i] < daily_ema50_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals