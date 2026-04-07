#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: Uses daily Donchian channel breakout with weekly EMA trend filter and volume confirmation. 
Long when price breaks above upper Donchian(20) with volume > 1.5x 20-day average and price > weekly EMA50.
Short when price breaks below lower Donchian(20) with volume > 1.5x 20-day average and price < weekly EMA50.
Exit when price crosses the 10-day EMA in the opposite direction.
Designed to capture strong trends while filtering out choppy markets. Works in both bull and bear markets by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Daily Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily 10-period EMA for exit
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, min_periods=10, adjust=False).values
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    volume_s = pd.Series(volume)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_10[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below 10-day EMA
            if close[i] < ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above 10-day EMA
            if close[i] > ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            vol_ok = volume[i] > 1.5 * vol_ma_20[i]
            
            # Long: Break above upper Donchian with volume and above weekly EMA50
            if (high[i] > high_20[i] and  # Breakout condition
                vol_ok and
                close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Break below lower Donchian with volume and below weekly EMA50
            elif (low[i] < low_20[i] and  # Breakdown condition
                  vol_ok and
                  close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals