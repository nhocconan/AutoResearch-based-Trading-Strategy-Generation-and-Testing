#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above R3, 4h EMA50 up-trend, volume > 1.8x average, and in session (08-20 UTC)
# Short when price breaks below S3, 4h EMA50 down-trend, volume > 1.8x average, and in session
# Exit when price crosses the 50% level (midpoint between R3 and S3)
# Uses 4h for signal direction/trend, 1h only for entry timing and Camarilla levels.
# Discrete position sizing (0.20) targets 15-37 trades/year to avoid fee drag.
# Designed to work in both bull and bear markets by following the higher timeframe trend.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_Session_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla levels (based on previous period)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels using previous period
    high_prev = df_1h['high'].shift(1).values
    low_prev = df_1h['low'].shift(1).values
    close_prev = df_1h['close'].shift(1).values
    range_prev = high_prev - low_prev
    
    camarilla_r3 = close_prev + range_prev * 1.1 / 4
    camarilla_s3 = close_prev - range_prev * 1.1 / 4
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0
    
    # Align 1h indicators to 1h timeframe (no additional delay needed for Camarilla)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1h, camarilla_mid)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 4h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_mid = camarilla_mid_aligned[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 50% level (mean reversion)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above 50% level (mean reversion)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average (balanced filter)
            vol_confirmed = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above R3, 4h EMA50 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema50_4h and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3, 4h EMA50 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema50_4h and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals