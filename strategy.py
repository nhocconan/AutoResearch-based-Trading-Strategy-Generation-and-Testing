#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R1 AND price > 4h EMA50 AND volume > 1.8x 24-bar avg
# Short when price breaks below Camarilla S1 AND price < 4h EMA50 AND volume > 1.8x 24-bar avg
# Exit when price retests Camarilla pivot (central level)
# Uses discrete position sizing (0.20) to reduce fee drag and improve test generalization.
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to avoid overtrading.
# Uses 4h for signal direction (trend filter and Camarilla levels from 1d data), 1h only for entry timing.
# Session filter: 08-20 UTC to reduce noise trades.
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns
# with trend alignment preventing counter-trend trades.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for Camarilla levels (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_r1 = prev_close_1d + camarilla_range * 1.1 / 12.0
    camarilla_s1 = prev_close_1d - camarilla_range * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (they represent levels from previous 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: >1.8x 24-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    # Session filter: 08-20 UTC (already datetime64[ms], use .index.hour)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R1 AND price > 4h EMA50 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema50_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 4h EMA50 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema50_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals