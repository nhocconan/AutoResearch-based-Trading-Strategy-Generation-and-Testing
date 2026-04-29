#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Long when close > upper Donchian AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when close < lower Donchian AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit on opposite Donchian level touch (long exit at lower band, short exit at upper band)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-30 trades/year on 4h.
# Donchian channels provide clear breakout levels. 12h EMA50 filters counter-trend moves.
# Volume spike confirms breakout strength. Works in both bull and bear markets by following trends.

name = "4h_Donchian20_12hEMA50_VolumeBreakout_v1"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA50 needs 50 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_12h_aligned[i]
        upper_band = upper_donchian[i]
        lower_band = lower_donchian[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when close > upper Donchian AND price > 12h EMA50 AND volume confirmation
            if curr_close > upper_band and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when close < lower Donchian AND price < 12h EMA50 AND volume confirmation
            elif curr_close < lower_band and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when close < lower Donchian (opposite band)
            if curr_close < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when close > upper Donchian (opposite band)
            if curr_close > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals