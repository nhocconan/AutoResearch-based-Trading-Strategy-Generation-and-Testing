#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Hypothesis: Donchian(20) breakouts on 12h timeframe capture significant moves.
# Weekly trend filter (price vs 50-period SMA) ensures we trade with the dominant trend.
# Volume confirmation (1.5x 24-period average) filters breakouts with institutional participation.
# Works in both bull and break markets by following the weekly trend.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "12h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly 50-period SMA for trend filter
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_sma50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (trailing stop)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (trailing stop)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long breakout: price closes above Donchian high AND above weekly SMA50 (uptrend)
                if close[i] > donchian_high[i] and close[i] > weekly_sma50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below Donchian low AND below weekly SMA50 (downtrend)
                elif close[i] < donchian_low[i] and close[i] < weekly_sma50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals