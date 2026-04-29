#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when close > upper Donchian AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when close < lower Donchian AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit on opposite Donchian level touch (long exit at lower, short exit at upper)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 15-25 trades/year on 1d.
# Donchian provides clear structure, 1w EMA50 filters counter-trend moves, volume confirms participation.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
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
    
    # Calculate Donchian channels (20-period)
    # Upper = highest high over past 20 bars
    high_series = pd.Series(high)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    # Lower = lowest low over past 20 bars
    low_series = pd.Series(low)
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1w_aligned[i]
        upper = upper_donchian[i]
        lower = lower_donchian[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when close > upper AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when close < lower AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when close < lower (opposite level)
            if curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when close > upper (opposite level)
            if curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals