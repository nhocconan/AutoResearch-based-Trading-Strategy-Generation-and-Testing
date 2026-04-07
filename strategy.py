#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation
# Hypothesis: Donchian channel breakouts capture momentum moves. Using 4h trend filter
# ensures trades align with higher timeframe direction, reducing whipsaw. Volume
# confirmation ensures institutional participation. Works in bull markets (breakouts
# continue trends) and bear markets (breakouts reverse trends for mean reversion).
# Target: 15-37 trades/year (60-150 over 4 years) for 1h timeframe.

name = "1h_donchian_breakout_4h_trend_volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA for trend (21-period)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate rolling max/min
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    # Align to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 4h EMA or volume drops
            if (close[i] <= ema_4h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above 4h EMA or volume drops
            if (close[i] >= ema_4h_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above 1d Donchian high with volume and 4h uptrend
            if ((high[i] > donchian_high_aligned[i] or close[i] > donchian_high_aligned[i]) and
                vol_filter[i] and close[i] > ema_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below 1d Donchian low with volume and 4h downtrend
            elif ((low[i] < donchian_low_aligned[i] or close[i] < donchian_low_aligned[i]) and
                  vol_filter[i] and close[i] < ema_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals