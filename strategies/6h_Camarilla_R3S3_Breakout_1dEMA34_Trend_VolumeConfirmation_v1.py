#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter (price > 1d EMA34) and volume confirmation.
# Long when price breaks above R3 with 1d uptrend (price > 1d EMA34) and volume spike (>2x 20-bar avg).
# Short when price breaks below S3 with 1d downtrend (price < 1d EMA34) and volume spike.
# Exit when price returns to the Camarilla H3/L3 level (mean reversion to midpoint).
# Uses institutional Camarilla levels, 1d EMA34 for trend filter (more responsive than 50), and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 1d OHLC for Camarilla levels (completed 1d bar)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Align 1d data to 6h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    range_1d = prev_high_aligned - prev_low_aligned
    camarilla_r3 = prev_close_aligned + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_aligned - range_1d * 1.1 / 4
    camarilla_h3 = prev_close_aligned + range_1d * 1.1 / 6
    camarilla_l3 = prev_close_aligned - range_1d * 1.1 / 6
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and volume MA
    
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
            # Long: price breaks above R3, uptrend (price > 1d EMA34), volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, downtrend (price < 1d EMA34), volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to H3 (mean reversion)
            if curr_close <= curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to L3 (mean reversion)
            if curr_close >= curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals