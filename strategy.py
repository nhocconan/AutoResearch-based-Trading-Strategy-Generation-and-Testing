#!/usr/bin/env python3
"""
4h_Donchian20_12hTrend_Volume
Hypothesis: 4-hour Donchian channel breakouts with 12-hour trend filter and volume confirmation. Trades with the dominant 12h trend, entering on 20-period Donchian breakouts confirmed by volume surge. This captures medium-term momentum while avoiding counter-trend whipsaws. Works in bull markets via upward breakouts and in bear markets via downward breakouts, with volume filtering out false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    # Donchian channel (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_surge[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above 20-period high + 12h uptrend + volume surge
        long_entry = (close[i] > high_20[i] and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: price breaks below 20-period low + 12h downtrend + volume surge
        short_entry = (close[i] < low_20[i] and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit on opposite break with volume surge
        long_exit = close[i] < low_20[i] and volume_surge[i]
        short_exit = close[i] > high_20[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0