#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above 6h Camarilla R3 with 1d uptrend (price > 1d EMA34) and volume spike (>1.8x 20-bar avg).
# Short when price breaks below 6h Camarilla S3 with 1d downtrend (price < 1d EMA34) and volume spike.
# Exit when price returns to the 6h Camarilla midpoint (mean reversion).
# Uses institutional Camarilla structure, 1d EMA34 for trend filter, and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous 6h OHLC for Camarilla levels (completed 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    prev_high_6h = df_6h['high'].shift(1).values
    prev_low_6h = df_6h['low'].shift(1).values
    prev_close_6h = df_6h['close'].shift(1).values
    
    # Align 6h data to 6h timeframe (identity alignment but ensures completed bar)
    prev_high_aligned = align_htf_to_ltf(prices, df_6h, prev_high_6h)
    prev_low_aligned = align_htf_to_ltf(prices, df_6h, prev_low_6h)
    prev_close_aligned = align_htf_to_ltf(prices, df_6h, prev_close_6h)
    
    # Calculate Camarilla levels from previous completed 6h bar
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    camarilla_s3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    camarilla_mid = prev_close_aligned  # Midpoint is the close of previous bar
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_6h, camarilla_mid)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_mid = camarilla_mid_aligned[i]
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
            # Exit condition: price returns to midpoint (mean reversion)
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to midpoint (mean reversion)
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals