#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above 1h Camarilla R3 with 4h uptrend (price > 4h EMA50) and volume spike (>2.0x 24-bar avg).
# Short when price breaks below 1h Camarilla S3 with 4h downtrend (price < 4h EMA50) and volume spike.
# Exit when price returns to 1h Camarilla pivot point (mean reversion).
# Uses institutional Camarilla structure, 4h EMA50 for trend filter, and volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Session filter: 08-20 UTC to reduce noise trades.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for performance)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 1h OHLC for Camarilla calculation (completed 1h bar)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    prev_high_1h = df_1h['high'].shift(1).values
    prev_low_1h = df_1h['low'].shift(1).values
    prev_close_1h = df_1h['close'].shift(1).values
    
    # Align 1h data to 1h timeframe (identity alignment but ensures completed bar)
    prev_high_aligned = align_htf_to_ltf(prices, df_1h, prev_high_1h)
    prev_low_aligned = align_htf_to_ltf(prices, df_1h, prev_low_1h)
    prev_close_aligned = align_htf_to_ltf(prices, df_1h, prev_close_1h)
    
    # Calculate Camarilla levels from previous 1h bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Pivot = (H+L+C)/3
    # We use R3/S3 for breakout and pivot for exit
    cam_r3 = prev_close_aligned + ((prev_high_aligned - prev_low_aligned) * 1.1 / 4)
    cam_s3 = prev_close_aligned - ((prev_high_aligned - prev_low_aligned) * 1.1 / 4)
    cam_pivot = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3
    
    # Volume confirmation: volume > 2.0x 24-period average (more strict for 1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if indicators not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(cam_r3[i]) or np.isnan(cam_s3[i]) or 
            np.isnan(cam_pivot[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = cam_r3[i]
        curr_s3 = cam_s3[i]
        curr_pivot = cam_pivot[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, uptrend (price > 4h EMA50), volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_50_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, downtrend (price < 4h EMA50), volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_50_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to pivot point (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price returns to pivot point (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals