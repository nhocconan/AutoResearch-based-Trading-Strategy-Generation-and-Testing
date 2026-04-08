#!/usr/bin/env python3
"""
1d weekly Donchian breakout with 1w trend filter and volume confirmation
Hypothesis: Price breaking weekly Donchian(50) channels in direction of weekly EMA(21) trend with volume surge captures momentum. Weekly timeframe filters noise and reduces trade frequency. Only takes long in uptrend, short in downtrend. Target: 15-30 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Weekly volume filter: current volume > 1.5x 20-period average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_surge_1w = volume_1w > (vol_ma_1w * 1.5)
    vol_surge_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_surge_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_surge_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price breaks below weekly Donchian(20) low
            donchian_low = np.min(low_1w[max(0, i-20):i+1]) if len(low_1w[max(0, i-20):i+1]) > 0 else low_1w[i]
            if (close[i] <= ema_21_1w_aligned[i] or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price breaks above weekly Donchian(20) high
            donchian_high = np.max(high_1w[max(0, i-20):i+1]) if len(high_1w[max(0, i-20):i+1]) > 0 else high_1w[i]
            if (close[i] >= ema_21_1w_aligned[i] or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Weekly Donchian(50) channels
            donchian_high = np.max(high_1w[max(0, i-50):i]) if len(high_1w[max(0, i-50):i]) > 0 else high_1w[i]
            donchian_low = np.min(low_1w[max(0, i-50):i]) if len(low_1w[max(0, i-50):i]) > 0 else low_1w[i]
            
            # Long: price breaks above weekly Donchian(50) high + volume surge + uptrend
            if (close[i] > donchian_high and
                close[i] > ema_21_1w_aligned[i] and
                vol_surge_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian(50) low + volume surge + downtrend
            elif (close[i] < donchian_low and
                  close[i] < ema_21_1w_aligned[i] and
                  vol_surge_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals