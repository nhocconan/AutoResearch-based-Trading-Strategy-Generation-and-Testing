#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when price reverts to Donchian middle (mean reversion) or opposite band touch
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide clear trend structure, 1d EMA50 filters counter-trend moves in bear markets,
# volume confirmation reduces false breakouts. This combination has proven effective on ETH/SOL.

name = "4h_Donchian20_1dEMA50_VolumeConfirm_Trend_v1"
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
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_middle = donchian_middle[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Donchian middle OR touches lower band
            if curr_close <= curr_middle or curr_close <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price reverts to Donchian middle OR touches upper band
            if curr_close >= curr_middle or curr_close >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1d EMA50 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d EMA50 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals