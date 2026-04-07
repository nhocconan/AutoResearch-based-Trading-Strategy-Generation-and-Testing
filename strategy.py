#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 1D Volume + Trend Filter
# Hypothesis: 4-hour Donchian channel breakouts capture momentum while
# 1-day volume surge confirms institutional participation. 1-day EMA filter
# ensures trades align with higher-timeframe trend, reducing whipsaws in
# ranging markets. Target: 25-50 trades/year to minimize fee drag on 4h.
name = "4h_donchian_breakout_1d_volume_trend_v1"
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
    
    # Get 1-day data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day volume average (20-period)
    daily_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20_4h[i]) or np.isnan(daily_ema_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or trend turns bearish
            if close[i] < low_20[i] or close[i] < daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or trend turns bullish
            if close[i] > high_20[i] or close[i] > daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume surge (>2x 20-day average) and trend alignment
            if volume[i] > (vol_ma_20_4h[i] * 2.0):
                # Breakout long: price breaks above upper band with uptrend
                if close[i] > high_20[i] and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below lower band with downtrend
                elif close[i] < low_20[i] and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals