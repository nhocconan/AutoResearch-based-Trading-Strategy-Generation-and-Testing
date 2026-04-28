#!/usr/bin/env python3
"""
4h_Donchian_Breakout_1dTrend_Volume_Confirmation
Hypothesis: 4h Donchian(20) breakout aligned with 1d trend (EMA50) and volume confirmation.
Works in bull/bear markets by filtering breakouts with trend and volume to reduce false signals.
Targets 20-50 trades/year with tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    d_uptrend = close > ema_50_1d_aligned
    d_downtrend = close < ema_50_1d_aligned
    
    # 4h Donchian(20) - calculate on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian channels on 4h: upper = max(high, 20), lower = min(low, 20)
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above Donchian high + daily uptrend + volume surge
        long_entry = (close[i] > donch_high_4h_aligned[i] and 
                     d_uptrend[i] and 
                     volume_surge[i])
        
        # Short: price breaks below Donchian low + daily downtrend + volume surge
        short_entry = (close[i] < donch_low_4h_aligned[i] and 
                      d_downtrend[i] and 
                      volume_surge[i])
        
        # Exit on opposite Donchian level break with volume surge
        long_exit = close[i] < donch_low_4h_aligned[i] and volume_surge[i]
        short_exit = close[i] > donch_high_4h_aligned[i] and volume_surge[i]
        
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

name = "4h_Donchian_Breakout_1dTrend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0