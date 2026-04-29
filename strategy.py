#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below Camarilla S1 AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when price retests Camarilla pivot (central level)
# Uses discrete position sizing (0.30) to balance return and risk. Target: 20-50 trades/year on 4h timeframe.
# Combines intraday support/resistance (Camarilla) with HTF trend filter and volume confirmation
# to capture strong breakouts while avoiding false signals in choppy markets.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike_v1"
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
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_r1 = prev_close_1d + camarilla_range * 1.1 / 2.0
    camarilla_s1 = prev_close_1d - camarilla_range * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (they represent levels from previous 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
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
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R1 AND price > 12h EMA50 AND volume confirmation
            if curr_close > curr_r1 and curr_close > curr_ema50_12h and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < curr_s1 and curr_close < curr_ema50_12h and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals