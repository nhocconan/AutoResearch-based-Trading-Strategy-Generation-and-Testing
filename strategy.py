#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1w EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below S3 AND close < 1w EMA34 AND volume > 1.8x 20-bar avg
# Exit when price retouches the central pivot (P) level
# Uses discrete position sizing (0.25) to reduce fee drag while maintaining profitability.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.
# R3/S3 levels provide strong breakout confirmation with moderate frequency.
# 1w EMA34 filters counter-trend moves effectively in both bull and bear markets.
# Volume confirmation ensures institutional participation, reducing whipsaws.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm_v1"
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
    
    # Get 1w data for EMA34 trend filter and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous week's OHLC
    # Camarilla: P = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/4
    # S3 = C - (H - L) * 1.1/4
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    p = (high_1w + low_1w + close_1w) / 3.0
    # R3 and S3 levels
    r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (using previous week's levels)
    p_aligned = align_htf_to_ltf(prices, df_1w, p)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA warmup and EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(p_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1w_aligned[i]
        curr_p = p_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retouches central pivot P
            if curr_close <= curr_p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches central pivot P
            if curr_close >= curr_p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1w EMA34 AND volume confirmation
            if curr_close > curr_r3 and close[i] > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND close < 1w EMA34 AND volume confirmation
            elif curr_close < curr_s3 and close[i] < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals