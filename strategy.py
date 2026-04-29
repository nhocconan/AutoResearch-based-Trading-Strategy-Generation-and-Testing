#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 20-bar high AND close > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below 20-bar low AND close < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses 1d EMA50 (trend change)
# Uses discrete position sizing (0.30) for balanced risk/return.
# Donchian channels provide clear structure, 1d EMA50 filters for longer-term trend alignment,
# and high volume confirmation ensures institutional participation.
# Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by following the 1d EMA50 trend with precise breakout entries.

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Calculate Donchian channels (20-bar high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA50 (trend change)
            if curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA50 (trend change)
            if curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND close > 1d EMA50 AND volume confirmation
            if curr_close > curr_donch_high and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian low AND close < 1d EMA50 AND volume confirmation
            elif curr_close < curr_donch_low and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals