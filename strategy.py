#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND close > 1d EMA50
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND close < 1d EMA50
# Exit when price retests Donchian(20) midpoint or opposite breakout level
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 19-50 trades/year on 4h timeframe.
# Donchian provides clear structure, volume confirms breakout strength, 1d EMA50 filters counter-trend moves.
# Works in bull via breakout continuation, in bear via breakdown continuation.

name = "4h_Donchian20_VolumeConfirm_1dEMA50_Trend_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_dch_high = donchian_high[i]
        curr_dch_low = donchian_low[i]
        curr_dch_mid = donchian_mid[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint or breaks below Donchian low
            if curr_low <= curr_dch_mid or curr_close <= curr_dch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint or breaks above Donchian high
            if curr_high >= curr_dch_mid or curr_close >= curr_dch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation AND close > 1d EMA50
            if curr_high > curr_dch_high and vol_conf and curr_close > curr_ema50_1d:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND volume confirmation AND close < 1d EMA50
            elif curr_low < curr_dch_low and vol_conf and curr_close < curr_ema50_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals