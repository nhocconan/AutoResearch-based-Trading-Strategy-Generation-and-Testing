#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Donchian upper AND close > 1d EMA34 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower AND close < 1d EMA34 AND volume > 1.5x 20-bar avg
# Exit when price retraces to 50% of the Donchian channel width
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
# Target: 100-180 total trades over 4 years (25-45/year) on 4h.
# Donchian channels provide objective breakout levels, 1d EMA34 filters trend direction,
# volume confirmation ensures breakout validity. Works in bull markets (breakouts with volume)
# and bear markets (breakdowns with volume) by capturing strong directional moves.

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v2"
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
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    donchian_width = donchian_upper - donchian_lower
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Donchian and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        width = donchian_width[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retraces to 50% of Donchian channel (middle level)
            if curr_close <= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retraces to 50% of Donchian channel (middle level)
            if curr_close >= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND close > 1d EMA34 AND volume confirmation
            if curr_close > upper and close[i] > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND close < 1d EMA34 AND volume confirmation
            elif curr_close < lower and close[i] < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals