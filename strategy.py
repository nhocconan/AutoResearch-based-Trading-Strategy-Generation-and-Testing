#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > weekly EMA50 AND volume > 2.0x 20-day avg volume
# Short when price breaks below 20-day low AND price < weekly EMA50 AND volume > 2.0x 20-day avg volume
# Exit when price crosses the opposite Donchian level (mean reversion to median)
# Uses discrete position sizing (0.30) to balance return and fee drag.
# Target: 20-50 trades/year on 1d (80-200 total over 4 years).
# Donchian levels provide clear breakout points; weekly EMA50 filters counter-trend moves.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in both bull and bear markets by allowing long/short entries based on trend filter.

name = "1d_Donchian20_1wEMA50_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Donchian channels (need prior day's data to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) using prior day's data only
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling window on prior data: shift by 1 to use only completed days
    high_shifted = np.roll(high_1d, 1)
    low_shifted = np.roll(low_1d, 1)
    high_shifted[0] = np.nan  # First value has no prior day
    low_shifted[0] = np.nan
    
    # Calculate 20-period rolling max/min on shifted data
    high_series = pd.Series(high_shifted)
    low_series = pd.Series(low_shifted)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (already aligned via get_htf_data)
    # No additional alignment needed as df_1d is already daily
    
    # Volume confirmation: >2.0x 20-day average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1w_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian band (mean reversion)
            if curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian band (mean reversion)
            if curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND price > weekly EMA50 AND volume confirmation
            if curr_close > upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower Donchian AND price < weekly EMA50 AND volume confirmation
            elif curr_close < lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals