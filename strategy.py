#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA50 trend filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND price > 12h EMA50
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND price < 12h EMA50
# Exit when price crosses Donchian(10) opposite level (e.g., long exits at Donchian(10) low)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 20-40 trades/year per symbol.
# Donchian provides objective structure, volume confirms breakout strength, 12h EMA50 filters counter-trend moves.

name = "4h_Donchian20_VolumeConfirm_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian channels (10-period) for exits
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian(20) and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_donchian_high_20 = donchian_high_20[i]
        curr_donchian_low_20 = donchian_low_20[i]
        curr_donchian_high_10 = donchian_high_10[i]
        curr_donchian_low_10 = donchian_low_10[i]
        curr_ema50 = ema_50_12h_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian(10) low
            if curr_close < curr_donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(10) high
            if curr_close > curr_donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian(20) high 
            # AND volume confirmation AND price > 12h EMA50
            if curr_close > curr_donchian_high_20 and vol_conf and curr_close > curr_ema50:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian(20) low
            # AND volume confirmation AND price < 12h EMA50
            elif curr_close < curr_donchian_low_20 and vol_conf and curr_close < curr_ema50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals