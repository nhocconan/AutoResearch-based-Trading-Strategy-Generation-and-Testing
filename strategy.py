#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with 1w uptrend (close > 1w EMA34) and volume > 1.8x 24-bar avg.
# Short when price breaks below Camarilla S3 with 1w downtrend (close < 1w EMA34) and volume > 1.8x 24-bar avg.
# Exit on opposite Camarilla level (S3 for long, R3 for short) or close below/above 1w EMA34.
# Uses proven Camarilla structure from 4h/6h winners adapted to 12h timeframe with stricter volume confirmation (1.8x) and 1w EMA34 trend filter.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag while capturing major trend moves.
# 1w EMA34 provides robust long-term trend filter, reducing false signals in choppy markets and bear rallies.

name = "12h_Camarilla_R3S3_1wEMA34_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Previous 1d OHLC for completed 1d bar (no look-ahead) - for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Align 1d data to 12h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Camarilla levels from previous completed 1d bar (no look-ahead)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    rng = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + 1.125 * rng
    camarilla_s3 = prev_close_aligned - 1.125 * rng
    
    # Volume confirmation: volume > 1.8x 24-period average (24*12h = 12 days, strict to avoid overtrading)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, uptrend (close > 1w EMA34), volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, downtrend (close < 1w EMA34), volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price touches Camarilla S3 OR close below 1w EMA34
            if (curr_close <= curr_s3 or 
                curr_close < curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price touches Camarilla R3 OR close above 1w EMA34
            if (curr_close >= curr_r3 or 
                curr_close > curr_ema_34_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals