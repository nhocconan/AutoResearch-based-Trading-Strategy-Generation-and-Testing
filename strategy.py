#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price reverts to 20-day midpoint (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 10-25 trades/year on 1d timeframe.
# Donchian channels provide robust trend structure, 1w EMA50 filters counter-trend moves in bear markets,
# volume confirmation ensures breakout authenticity. Works in both bull (trend continuation) and bear (mean reversion during rallies).

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) from previous 20 days (using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period rolling high/low on daily data
    rolling_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    rolling_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Donchian levels from previous bar (to avoid look-ahead)
    prev_high_20 = np.roll(rolling_high, 1)
    prev_low_20 = np.roll(rolling_low, 1)
    prev_high_20[0] = high_1d[0]  # first bar
    prev_low_20[0] = low_1d[0]    # first bar
    
    # Donchian midpoint for exit
    prev_mid_20 = (prev_high_20 + prev_low_20) / 2.0
    
    # Align Donchian levels to 1d timeframe (already daily, but using align for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, prev_mid_20)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_1w_aligned[i]
        curr_high = donchian_high_aligned[i]
        curr_low = donchian_low_aligned[i]
        curr_mid = donchian_mid_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 1w EMA50 AND volume confirmation
            if curr_close > curr_high and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < 1w EMA50 AND volume confirmation
            elif curr_close < curr_low and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals