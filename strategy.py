#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses Donchian(10) midpoint OR trend reverses (price crosses 1w EMA50)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Donchian breakouts capture strong momentum, 1w EMA50 filters counter-trend moves,
# volume confirmation ensures follow-through. Works in bull markets (upside breakouts) 
# and bear markets (downside breakouts) with proper trend alignment.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v2"
timeframe = "1d"
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
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = ((high_series.rolling(window=10, min_periods=10).max() + 
                       low_series.rolling(window=10, min_periods=10).min()) / 2).values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 50)  # Donchian(20), volume MA, EMA50 alignment warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_dch_high = donchian_high_20[i]
        curr_dch_low = donchian_low_20[i]
        curr_dch_mid = donchian_mid_10[i]
        curr_ema50 = ema_50_1w_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian(10) midpoint OR price crosses below 1w EMA50
            if curr_close < curr_dch_mid or curr_close < curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(10) midpoint OR price crosses above 1w EMA50
            if curr_close > curr_dch_mid or curr_close > curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume confirmation
            if curr_high > curr_dch_high and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume confirmation
            elif curr_low < curr_dch_low and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals