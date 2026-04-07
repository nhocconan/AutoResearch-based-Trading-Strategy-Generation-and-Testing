#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian 20 with Volume and Trend Filter
# Hypothesis: Donchian(20) breakouts on 4h with daily trend filter (price vs 200 EMA)
# and volume confirmation work in both bull and bear markets by capturing
# momentum bursts while avoiding false breakouts in low-volume or counter-trend conditions.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_donchian20_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Daily close for EMA calculation
    daily_close = df_daily['close'].values
    # Calculate 200 EMA on daily timeframe
    daily_close_series = pd.Series(daily_close)
    ema_200_daily = daily_close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    # Align to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_daily, ema_200_daily)
    
    # Donchian(20) channels on 4h data
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
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below Donchian low or trend/volume filter fails
            if (close[i] < donchian_low[i] or
                close[i] < ema_200_aligned[i] or
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: close above Donchian high or trend/volume filter fails
            if (close[i] > donchian_high[i] or
                close[i] > ema_200_aligned[i] or
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with uptrend and volume
            if (close[i] > donchian_high[i] and
                close[i] > ema_200_aligned[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakout below Donchian low with downtrend and volume
            elif (close[i] < donchian_low[i] and
                  close[i] < ema_200_aligned[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals