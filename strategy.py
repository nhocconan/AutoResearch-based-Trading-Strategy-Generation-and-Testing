#!/usr/bin/env python3
"""
12h_donchian_20_1d_trend_volume_v1
Hypothesis: On 12-hour timeframe, use Donchian channel breakout from 20-period lookback with 1-day trend filter and volume confirmation. 
Enter long when price breaks above 20-period high with 1-day EMA uptrend and volume > 1.5x average, short when price breaks below 20-period low with 1-day EMA downtrend and volume > 1.5x average. 
Exit when price crosses 10-period EMA in opposite direction. Designed for low frequency (12-37 trades/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on daily close for trend filter
    d_close = df_1d['close'].values
    ema_20d = pd.Series(d_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20d_aligned = align_htf_to_ltf(prices, df_1d, ema_20d)
    
    # Calculate Donchian channels (20-period high/low) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit signal
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if daily EMA not available
        if np.isnan(ema_20d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price crosses below 10-period EMA
            if close[i] < ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit when price crosses above 10-period EMA
            if close[i] > ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with 1-day EMA uptrend and volume confirmation
            long_entry = (close[i] > donchian_high[i]) and (ema_20d_aligned[i] > ema_20d_aligned[i-1]) and vol_confirm
            # Short entry: price breaks below Donchian low with 1-day EMA downtrend and volume confirmation
            short_entry = (close[i] < donchian_low[i]) and (ema_20d_aligned[i] < ema_20d_aligned[i-1]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
    
    return signals