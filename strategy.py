#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v5
Hypothesis: On 4-hour timeframe, use Donchian breakout with 1-day trend filter and volume confirmation. 
Enter long when price breaks above 20-period high with volume > 1.8x 50-period average and price > 50 EMA.
Enter short when price breaks below 20-period low with volume > 1.8x 50-period average and price < 50 EMA.
Exit when price crosses 50 EMA in opposite direction. Designed for low frequency (20-50 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v5"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50 EMA for daily trend
    d_close = df_1d['close'].values
    ema_50_1d = pd.Series(d_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 50-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 50 EMA for exit signal
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if daily trend data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 50-period average
        vol_confirm = volume[i] > 1.8 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price crosses below 50 EMA
            if close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above 50 EMA
            if close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation AND price > 50 EMA (uptrend)
            long_entry = (high[i] > donchian_high[i]) and vol_confirm and (close[i] > ema_50[i])
            # Short entry: price breaks below Donchian low with volume confirmation AND price < 50 EMA (downtrend)
            short_entry = (low[i] < donchian_low[i]) and vol_confirm and (close[i] < ema_50[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals