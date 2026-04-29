#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when close > upper Donchian(20) AND price > 1d EMA50 AND volume > 2x 20-bar avg
# Short when close < lower Donchian(20) AND price < 1d EMA50 AND volume > 2x 20-bar avg
# Exit when price crosses opposite Donchian band OR momentum fades (volume < 1.5x avg)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.
# Donchian channels provide clear breakout levels, 1d EMA50 filters counter-trend moves,
# volume confirmation ensures institutional participation. Works in both bull/bear markets.

name = "12h_Donchian20_1dEMA50_VolumeConfirm_v1"
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
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donch = high_series.rolling(window=20, min_periods=20).max().values
    lower_donch = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    volume_exit = volume < 1.5 * volume_ma_20  # exit when volume fades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        vol_exit_cond = volume_exit[i]
        ema_50 = ema_50_1d_aligned[i]
        curr_close = close[i]
        curr_upper = upper_donch[i]
        curr_lower = lower_donch[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian OR volume fades
            if curr_close < curr_lower or vol_exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian OR volume fades
            if curr_close > curr_upper or vol_exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when close > upper Donchian AND price > 1d EMA50 AND volume confirmation
            if curr_close > curr_upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when close < lower Donchian AND price < 1d EMA50 AND volume confirmation
            elif curr_close < curr_lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals