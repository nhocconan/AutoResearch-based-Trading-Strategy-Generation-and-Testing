#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period high, close > 1w EMA50, and volume > 1.8x 20-bar avg.
# Short when price breaks below 20-period low, close < 1w EMA50, and volume > 1.8x 20-bar avg.
# Exit when price crosses the 10-period EMA in the opposite direction.
# Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Donchian channels provide clear breakout levels based on price action.
# 1w EMA50 filters for higher timeframe trend alignment to avoid counter-trend trades.
# Volume confirmation reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.
# Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for 1w EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(ema_10[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high_ma_20 = high_ma_20[i]
        curr_low_ma_20 = low_ma_20[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_ema_10 = ema_10[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-period high, close > 1w EMA50, volume spike
            if (curr_close > curr_high_ma_20 and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low, close < 1w EMA50, volume spike
            elif (curr_close < curr_low_ma_20 and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below 10-period EMA
            if curr_close < curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above 10-period EMA
            if curr_close > curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals