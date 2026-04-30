#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band, close > 1d EMA50, and volume > 2.0x 20-bar avg.
# Short when price breaks below Donchian lower band, close < 1d EMA50, and volume > 2.0x 20-bar avg.
# Exit when price moves back inside the Donchian channel.
# Uses Donchian channel from 6h timeframe for structure.
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation with 2.0x threshold reduces false breakouts.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits.

name = "6h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 6h timeframe
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_upper = high_ma_20[i]
        curr_lower = low_ma_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper band, close > 1d EMA50, volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, close < 1d EMA50, volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back inside Donchian channel (below upper band)
            if curr_close < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves back inside Donchian channel (above lower band)
            if curr_close > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals