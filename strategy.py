#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1D Trend and Volume Confirmation
# Hypothesis: Donchian channel breakouts on 12h timeframe capture significant trends
# while avoiding whipsaw. We filter trades in direction of 1-day EMA(50) trend
# and require volume confirmation. This combination works in both bull and bear
# markets by capturing strong directional moves while avoiding false breakouts
# in ranging conditions. Target: 15-30 trades/year to minimize fee drag.
name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Donchian Channel (20-period) on 12h timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or trend turns bearish
            if close[i] < low_20[i] or close[i] < daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or trend turns bullish
            if close[i] > high_20[i] or close[i] > daily_ema_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above upper Donchian band + above 1D EMA
                if close[i] > high_20[i] and close[i] > daily_ema_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band + below 1D EMA
                elif close[i] < low_20[i] and close[i] < daily_ema_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals