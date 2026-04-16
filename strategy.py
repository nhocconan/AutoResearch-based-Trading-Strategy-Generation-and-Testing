#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme readings with 1w EMA trend filter and volume confirmation.
# Long when 1w EMA rising, 1d Williams %R < -80 (oversold), and volume > 1.5x 20-period average.
# Short when 1w EMA falling, 1d Williams %R > -20 (overbought), and volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Williams %R captures mean reversion extremes,
# 1w EMA ensures alignment with major trend, volume confirms conviction.
# Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Get 1w data once before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_1w_dir = np.zeros_like(ema_30_1w)
    ema_1w_dir[1:] = np.where(ema_30_1w[1:] > ema_30_1w[:-1], 1, np.where(ema_30_1w[1:] < ema_30_1w[:-1], -1, 0))
    
    # Get 1h data once before loop for volume MA
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    vol_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_1w_dir_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_dir)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20_1h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_1w_dir_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        williams_val = williams_r_aligned[i]
        ema_dir_val = ema_1w_dir_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R rises above -50 (exit overbought) or EMA trend turns down
            if williams_val > -50 or ema_dir_val <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R falls below -50 (exit oversold) or EMA trend turns up
            if williams_val < -50 or ema_dir_val >= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average (1h)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # LONG: 1w EMA up, Williams %R oversold (< -80), volume confirmation
            if (ema_dir_val > 0) and (williams_val < -80) and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: 1w EMA down, Williams %R overbought (> -20), volume confirmation
            elif (ema_dir_val < 0) and (williams_val > -20) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dWilliamsR_1wEMA_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0