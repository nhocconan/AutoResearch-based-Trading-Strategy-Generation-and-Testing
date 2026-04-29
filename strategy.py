#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
# Donchian breakouts capture momentum; EMA(50) filters for higher-timeframe trend alignment
# Volume confirmation ensures breakout validity. Works in both bull/bear by trading
# with the 1d trend. Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1d EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above upper Donchian + above 1d EMA50 + volume
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_50 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower Donchian + below 1d EMA50 + volume
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_50 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit on Donchian midpoint or trend change
            # Exit when price crosses below Donchian midpoint OR closes below 1d EMA50
            donchian_mid = (curr_highest_high + curr_lowest_low) / 2.0
            if curr_close < donchian_mid or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit on Donchian midpoint or trend change
            # Exit when price crosses above Donchian midpoint OR closes above 1d EMA50
            donchian_mid = (curr_highest_high + curr_lowest_low) / 2.0
            if curr_close > donchian_mid or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals