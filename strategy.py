#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: On 4h timeframe, buy when price breaks above 20-period Donchian high with 1d EMA uptrend and volume confirmation; sell when price breaks below 20-period Donchian low with 1d EMA downtrend and volume confirmation. Uses Donchian channels for breakout signals, 1d EMA for trend filter, and volume spike for confirmation. Designed for 75-200 total trades over 4 years to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA trend (50-period)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(lookback, 20, 50), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price breaks above Donchian high with 1d uptrend
                if (close[i] > donchian_high[i] and 
                    close[i-1] <= donchian_high[i-1] and
                    close[i] > ema_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.30
                # Bearish breakout: price breaks below Donchian low with 1d downtrend
                elif (close[i] < donchian_low[i] and 
                      close[i-1] >= donchian_low[i-1] and
                      close[i] < ema_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.30
    
    return signals