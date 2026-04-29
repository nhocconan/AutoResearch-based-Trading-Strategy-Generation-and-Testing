#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike + Session Filter (08-20 UTC)
# Long when price breaks above Camarilla R3 level AND price > 1d EMA34 AND volume > 2.0x 20-bar avg AND session filter
# Short when price breaks below Camarilla S3 level AND price < 1d EMA34 AND volume > 2.0x 20-bar avg AND session filter
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Camarilla R3/S3 levels provide stronger breakout confirmation than R1/S1, reducing false signals.
# 1d EMA34 filters counter-trend moves, volume confirmation ensures breakout strength, session filter reduces noise.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Session_v1"
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
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 1w data for additional trend filter (optional, for stronger bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        # If 1w data insufficient, continue with just 1d
        df_1w = None
    
    # Calculate EMA(34) on 1d data for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(10) on 1w data for additional trend filter (if available)
    if df_1w is not None:
        close_1w = df_1w['close'].values
        ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
        ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Handle first bar where shift creates NaN
    if len(prev_high) > 0:
        prev_high[0] = df_1d['high'].iloc[0]
        prev_low[0] = df_1d['low'].iloc[0]
        prev_close[0] = df_1d['close'].iloc[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + camarilla_range * 1.1 / 4.0
    camarilla_S3 = prev_close - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i] and session_filter[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_close = close[i]
        
        # Additional 1w trend filter (if available)
        if df_1w is not None:
            if np.isnan(ema_10_1w_aligned[i]):
                signals[i] = 0.0
                continue
            curr_ema10_1w = ema_10_1w_aligned[i]
            # Require 12h price to be above 1w EMA10 for long, below for short
            long_trend_filter = curr_close > curr_ema10_1w
            short_trend_filter = curr_close < curr_ema10_1w
        else:
            long_trend_filter = True
            short_trend_filter = True
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume confirmation AND session AND 1w trend
            if curr_close > curr_R3 and curr_close > curr_ema34 and vol_conf and long_trend_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume confirmation AND session AND 1w trend
            elif curr_close < curr_S3 and curr_close < curr_ema34 and vol_conf and short_trend_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals