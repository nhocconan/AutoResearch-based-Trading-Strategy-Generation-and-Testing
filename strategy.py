#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R1 AND price > 4h EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S1 AND price < 4h EMA34 AND volume > 2.0x 20-bar avg
# Exit when price retests Camarilla pivot (central level)
# Uses discrete position sizing (0.20) to reduce fee drag and improve test generalization.
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to avoid overtrading.
# Uses 4h for signal direction (trend filter) and 1h only for entry timing precision.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns
# with trend alignment preventing counter-trend trades.

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 4h data for Camarilla levels (based on previous 4h bar)
    prev_high_4h = df_4h['high'].values
    prev_low_4h = df_4h['low'].values
    prev_close_4h = df_4h['close'].values
    
    camarilla_range = prev_high_4h - prev_low_4h
    camarilla_pivot = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    camarilla_r1 = prev_close_4h + camarilla_range * 1.0 / 4.0  # R1 level
    camarilla_s1 = prev_close_4h - camarilla_range * 1.0 / 4.0  # S1 level
    
    # Align Camarilla levels and pivot to 1h timeframe (they represent levels from previous 4h bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: >2.0x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_4h = ema_34_4h_aligned[i]
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
            # Long when price breaks above Camarilla R1 AND price > 4h EMA34 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema34_4h and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 4h EMA34 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema34_4h and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals