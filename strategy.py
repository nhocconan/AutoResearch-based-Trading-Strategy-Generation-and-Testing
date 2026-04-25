#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Trade 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation. 
In bull markets: long when price breaks above 20-period high and price > 1d EMA50. 
In bear markets: short when price breaks below 20-period low and price < 1d EMA50. 
Requires volume > 1.3x 20-period average for confirmation. 
Exit on opposite Donchian breakout or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Donchian channels provide objective breakout levels that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate 12h Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        highest_high = np.max(high[lookback_start:i+1])
        lowest_low = np.min(low[lookback_start:i+1])
        
        # Determine 1d HTF trend (bullish = price above daily EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above 20-period high + 1d uptrend + volume confirmation
            long_setup = (close[i] > highest_high) and htf_1d_bullish and volume_confirm
            
            # Short setup: price breaks below 20-period low + 1d downtrend + volume confirmation
            short_setup = (close[i] < lowest_low) and htf_1d_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below 20-period low OR 1d trend turns bearish
            if (close[i] < lowest_low) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above 20-period high OR 1d trend turns bullish
            if (close[i] > highest_high) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0