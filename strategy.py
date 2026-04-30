#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w Supertrend trend filter and volume confirmation.
# Williams %R(14) identifies overbought/oversold conditions. Long when %R < -80 (oversold) with 1w uptrend (Supertrend bullish) and volume spike (>1.5x 20-bar avg).
# Short when %R > -20 (overbought) with 1w downtrend (Supertrend bearish) and volume spike.
# Exit when %R crosses -50 (mean reversion to midpoint) or opposite extreme (%R > -80 for longs, %R < -20 for shorts).
# Uses proven mean reversion logic with strict volume confirmation to limit trades (target 30-100 total trades over 4 years).
# 1d timeframe reduces fee drag while 1w Supertrend ensures alignment with major trend, working in both bull and bear markets.

name = "1d_WilliamsR14_1wSupertrend_Trend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter (ATR=10, mult=3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], 0.0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_1w = (high_1w + low_1w) / 2.0
    upperband_1w = hl2_1w + (3.0 * atr_1w)
    lowerband_1w = hl2_1w - (3.0 * atr_1w)
    
    supertrend_1w = np.zeros_like(close_1w)
    direction_1w = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend_1w[0] = upperband_1w[0]
    direction_1w[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend_1w[i-1]:
            direction_1w[i] = 1
        else:
            direction_1w[i] = -1
        
        if direction_1w[i] == 1:
            supertrend_1w[i] = max(lowerband_1w[i], supertrend_1w[i-1])
        else:
            supertrend_1w[i] = min(upperband_1w[i], supertrend_1w[i-1])
    
    # Align 1w Supertrend direction to 1d timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # Williams %R(14) on 1d
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_williams_r = williams_r[i]
        curr_supertrend_dir = supertrend_dir_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), 1w uptrend (Supertrend bullish), volume spike
            if (curr_williams_r < -80.0 and 
                curr_supertrend_dir > 0 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), 1w downtrend (Supertrend bearish), volume spike
            elif (curr_williams_r > -20.0 and 
                  curr_supertrend_dir < 0 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R crosses -50 (mean reversion) or rises above -80 (exit oversold)
            if curr_williams_r > -50.0 or curr_williams_r > -80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R crosses -50 (mean reversion) or falls below -20 (exit overbought)
            if curr_williams_r < -50.0 or curr_williams_r < -20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals