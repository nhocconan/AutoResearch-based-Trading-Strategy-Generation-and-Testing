#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND price > 12h EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND price < 12h EMA34 AND volume > 2.0x 20-bar avg
# Exit when price retests Camarilla pivot (central level) or 12h EMA34
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 19-50 trades/year on 4h timeframe.
# Combines intraday support/resistance (Camarilla) with HTF trend filter and volume confirmation
# to capture strong breakouts while avoiding false signals in choppy markets.

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike_v1"
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
    
    # Get 12h data for EMA34 trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: based on previous bar's high, low, close
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # Pivot = (high+low+close)/3
    prev_high_12h = df_12h['high'].values
    prev_low_12h = df_12h['low'].values
    prev_close_12h = df_12h['close'].values
    
    camarilla_range = prev_high_12h - prev_low_12h
    camarilla_pivot = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    camarilla_r3 = prev_close_12h + camarilla_range * 1.1 / 4.0
    camarilla_s3 = prev_close_12h - camarilla_range * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (they represent levels from previous 12h bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_12h = ema_34_12h_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Camarilla pivot OR price retests 12h EMA34 (weakening bullish momentum)
            if curr_close <= curr_pivot or curr_close <= curr_ema34_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Camarilla pivot OR price retests 12h EMA34 (weakening bearish momentum)
            if curr_close >= curr_pivot or curr_close >= curr_ema34_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 12h EMA34 AND volume confirmation
            if curr_close > curr_r3 and curr_close > curr_ema34_12h and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 12h EMA34 AND volume confirmation
            elif curr_close < curr_s3 and curr_close < curr_ema34_12h and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals