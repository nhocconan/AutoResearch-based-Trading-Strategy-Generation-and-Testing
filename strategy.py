#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when price retests Camarilla pivot (central level) or 1w EMA50
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 7-25 trades/year on 1d timeframe.
# Combines daily support/resistance (Camarilla) with weekly trend filter and volume confirmation
# to capture strong breakouts while avoiding false signals in choppy markets.
# Novelty: Uses 1d primary timeframe (proven keep rate 40%) with 1w EMA50 for HTF trend filter.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous bar's high, low, close
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # Pivot = (high+low+close)/3
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_r3 = prev_close_1d + camarilla_range * 1.1 / 4.0
    camarilla_s3 = prev_close_1d - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 1d timeframe (they represent levels from previous 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot OR price retests 1w EMA50 (weakening bullish momentum)
            if curr_close <= curr_pivot or curr_close <= curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot OR price retests 1w EMA50 (weakening bearish momentum)
            if curr_close >= curr_pivot or curr_close >= curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1w EMA50 AND volume confirmation
            if curr_close > curr_r3 and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1w EMA50 AND volume confirmation
            elif curr_close < curr_s3 and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals