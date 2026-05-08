#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    for i in range(20, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly volume ratio (current vs 20-period average)
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = np.divide(volume_1w, vol_ma_1w, out=np.ones_like(volume_1w), where=vol_ma_1w!=0)
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + above average volume + price above EMA34 (uptrend)
            if (close[i] > donchian_high_aligned[i] and 
                vol_ratio_1w_aligned[i] > 1.3 and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + above average volume + price below EMA34 (downtrend)
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_ratio_1w_aligned[i] > 1.3 and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below weekly Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above weekly Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals