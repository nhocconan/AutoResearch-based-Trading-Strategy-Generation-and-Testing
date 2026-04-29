#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2x 20-bar avg
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2x 20-bar avg
# Exit when price retouches the central pivot (P) level
# Uses discrete position sizing (0.30) to balance profit and fee drag.
# Target: 100-200 total trades over 4 years (25-50/year) on 4h.
# Camarilla pivots provide precise intraday support/resistance levels.
# 1d EMA34 filters counter-trend moves, volume confirmation ensures institutional participation.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: P = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    p = (high_1d + low_1d + close_1d) / 3.0
    # R3 and S3 levels
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    p_aligned = align_htf_to_ltf(prices, df_1d, p)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA warmup and EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(p_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_12h_aligned[i]
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
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price retouches central pivot P
            if curr_close >= curr_p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 12h EMA50 AND volume confirmation
            if curr_close > curr_r3 and close[i] > ema_50 and vol_conf:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below S3 AND close < 12h EMA50 AND volume confirmation
            elif curr_close < curr_s3 and close[i] < ema_50 and vol_conf:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals