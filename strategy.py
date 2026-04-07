#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout + 1d Trend + Volume Confirmation
# Hypothesis: Donchian breakouts capture trend momentum with clear entry/exit levels.
# In trending markets (price > 1d EMA50), breakouts above/below 20-period Donchian channels
# signal continuation. In ranging markets, we avoid false breakouts using volume filter
# (volume > 1.5x 20-period average) to ensure institutional participation. Works in both
# bull and bear markets by following the 1d trend direction. 6h timeframe reduces noise
# while capturing multi-day moves. Target: 12-37 trades/year (50-150 over 4 years).
name = "6h_donchian_breakout_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 6h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max()
    lower_channel = low_series.rolling(window=20, min_periods=20).min()
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian channel
            if close[i] < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian channel
            if close[i] > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Only trade in direction of 1d trend
                if close[i] > daily_ema_6h[i]:  # Uptrend
                    # Long: breakout above upper Donchian channel
                    if close[i] > upper_channel[i]:
                        position = 1
                        signals[i] = 0.25
                elif close[i] < daily_ema_6h[i]:  # Downtrend
                    # Short: breakdown below lower Donchian channel
                    if close[i] < lower_channel[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals