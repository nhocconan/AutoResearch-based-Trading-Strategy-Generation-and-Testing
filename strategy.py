#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above Camarilla R3 with 1d uptrend (price > 1d EMA34) and volume > 2x 20-bar avg.
# Short when price breaks below Camarilla S3 with 1d downtrend (price < 1d EMA34) and volume spike.
# Exit on touch of Camarilla H3/L3 levels (mean reversion within the inner quadrant).
# Uses proven Camarilla structure with strict volume confirmation to limit trades (target 50-150 total trades over 4 years).
# 6h timeframe balances trend capture with lower fee drag vs lower TFs; 1d EMA filter ensures alignment with daily trend.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous 1d OHLC for completed 1d bar (no look-ahead)
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Align 1d data to 6h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Camarilla: range = prev_high - prev_low
    # R3 = prev_close + 1.1 * range / 2
    # S3 = prev_close - 1.1 * range / 2
    # H3 = prev_close + 1.1 * range / 4
    # L3 = prev_close - 1.1 * range / 4
    prev_range = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + 1.1 * prev_range / 2
    camarilla_s3 = prev_close_aligned - 1.1 * prev_range / 2
    camarilla_h3 = prev_close_aligned + 1.1 * prev_range / 4
    camarilla_l3 = prev_close_aligned - 1.1 * prev_range / 4
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_h3 = camarilla_h3[i]
        curr_l3 = camarilla_l3[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, uptrend (price > 1d EMA34), volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, downtrend (price < 1d EMA34), volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Camarilla H3 (mean reversion)
            if curr_close >= curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Camarilla L3 (mean reversion)
            if curr_close <= curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals