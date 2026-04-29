#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper AND close > 1d EMA34 AND volume > 2x 20-bar avg
# Short when price breaks below Donchian lower AND close < 1d EMA34 AND volume > 2x 20-bar avg
# Exit when price retouches the Donchian midpoint
# Uses discrete position sizing (0.30) to balance profit and fee drag.
# Target: 100-200 total trades over 4 years (25-50/year) on 4h.
# Donchian channels provide robust trend-following structure.
# 1d EMA34 filters counter-trend moves, volume confirmation ensures institutional participation.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).

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
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Donchian calculation (using previous day's OHLC)
    # We need previous day's high/low, so we shift by 1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) from previous day's OHLC
    # Upper = max(high_1d over last 20 days)
    # Lower = min(low_1d over last 20 days)
    # Middle = (Upper + Lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe (using previous day's levels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA warmup and EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1d_aligned[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_middle = donchian_middle_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retouches Donchian middle
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retouches Donchian middle
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND close > 1d EMA34 AND volume confirmation
            if curr_close > curr_upper and close[i] > ema_34 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian lower AND close < 1d EMA34 AND volume confirmation
            elif curr_close < curr_lower and close[i] < ema_34 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals