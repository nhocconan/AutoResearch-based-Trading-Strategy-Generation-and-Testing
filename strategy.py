#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period high) AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower band (20-period low) AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price retests the opposite Donchian band (lower for long exit, upper for short exit)
# Uses discrete position sizing (0.30) to balance risk and reward. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide clear structural breakout levels. 1w EMA50 filters counter-trend moves on weekly timeframe,
# volume confirmation ensures breakout strength with follow-through. Works in bull via breakout continuation,
# in bear via breakdown continuation. Novelty: using weekly EMA50 as trend filter (slower, more reliable than daily).

name = "4h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian lower band
            if curr_low <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian upper band
            if curr_high >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND price > 1w EMA50 AND volume confirmation
            if curr_high > curr_upper and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower band AND price < 1w EMA50 AND volume confirmation
            elif curr_low < curr_lower and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals