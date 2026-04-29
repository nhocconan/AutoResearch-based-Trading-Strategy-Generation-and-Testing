#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R3 level AND price > 4h EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 level AND price < 4h EMA50 AND volume > 2.0x 20-bar avg
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses 1h timeframe for entry timing, 4h for trend direction and Camarilla levels (from prior 4h bar)
# Designed to generate 15-37 trades/year (60-150 over 4 years) with discrete sizing (0.20) to minimize fee drag
# Camarilla levels provide precise support/resistance, 4h EMA50 filters counter-trend moves,
# volume confirmation ensures breakout strength. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = open_time.dt.hour.values
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h data
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = df_4h['high'].shift(1).values  # Previous 4h bar high
    prev_low = df_4h['low'].shift(1).values    # Previous 4h bar low
    prev_close = df_4h['close'].shift(1).values # Previous 4h bar close
    
    # Handle first bar where shift creates NaN
    if len(prev_high) > 0:
        prev_high[0] = df_4h['high'].iloc[0]
        prev_low[0] = df_4h['low'].iloc[0]
        prev_close[0] = df_4h['close'].iloc[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + camarilla_range * 1.1 / 4.0
    camarilla_S3 = prev_close - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_4h_aligned[i]
        curr_R3 = camarilla_R3_aligned[i]
        curr_S3 = camarilla_S3_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 4h EMA50 AND volume confirmation
            if curr_close > curr_R3 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 4h EMA50 AND volume confirmation
            elif curr_close < curr_S3 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals