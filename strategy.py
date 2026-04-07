#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: On 4-hour timeframe, buy when price breaks above 20-period Donchian high with 1-day uptrend (price > EMA50) and volume confirmation; sell when price breaks below 20-period Donchian low with 1-day downtrend (price < EMA50) and volume confirmation. Exit on opposite Donchian break. Trend filter reduces whipsaw, volume confirms momentum, Donchian provides clear breakout levels. Target: 20-40 trades/year to minimize fee dust while capturing trends.
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
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_close = df_1d['close'].values
    
    # Calculate EMA50 on daily for trend filter
    ema50 = pd.Series(d_close).ewm(span=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Calculate Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if trend filter not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs daily EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.3
        
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low
            if low[i] <= donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if high[i] >= donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high AND uptrend AND volume confirmed
            long_entry = (high[i] >= donch_high[i]) and uptrend and vol_confirmed
            
            # Short entry: price breaks below Donchian low AND downtrend AND volume confirmed
            short_entry = (low[i] <= donch_low[i]) and downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals