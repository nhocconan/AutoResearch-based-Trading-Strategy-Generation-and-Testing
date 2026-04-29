#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-bar high) AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below Donchian lower (20-bar low) AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when price retraces to 10-bar EMA on 4h (adaptive trailing)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years) to avoid overtrading.
# Donchian channels provide objective breakout levels, 1d EMA34 filters for higher-timeframe trend alignment,
# and volume confirmation ensures breakout validity. Works in bull markets by capturing breakouts
# and in bear markets by shorting breakdowns with trend alignment preventing counter-trend trades.

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-bar) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-bar EMA for exit on 4h
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 10)  # Donchian, EMA34, and EMA10 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_10[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_ema10 = ema_10[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retraces to 10-bar EMA
            if curr_close <= curr_ema10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retraces to 10-bar EMA
            if curr_close >= curr_ema10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1d EMA34 AND volume confirmation
            if curr_close > curr_dc_upper and curr_close > curr_ema34_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d EMA34 AND volume confirmation
            elif curr_close < curr_dc_lower and curr_close < curr_ema34_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals