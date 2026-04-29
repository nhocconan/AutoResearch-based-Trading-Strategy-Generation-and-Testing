#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-bar high AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-bar low AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price retests opposite Donchian level (20-bar low for long, 20-bar high for short)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Donchian provides objective breakout levels, 1w EMA50 filters counter-trend moves,
# volume confirmation ensures breakout strength. Works in bull via breakout continuation,
# in bear via breakdown continuation. Novelty: applying Donchian to 12h with weekly trend filter.

name = "12h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-bar high/low) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian low (20-bar low)
            if curr_low <= curr_donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian high (20-bar high)
            if curr_high >= curr_donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 1w EMA50 AND volume confirmation
            if curr_high > curr_donch_high and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < 1w EMA50 AND volume confirmation
            elif curr_low < curr_donch_low and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals