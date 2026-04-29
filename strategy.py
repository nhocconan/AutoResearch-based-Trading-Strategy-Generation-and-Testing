#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from 1d timeframe for institutional breakout points
# 1d EMA50 provides strong trend filter to avoid counter-trend trades
# Volume > 2.0x average confirms institutional participation and reduces false breakouts
# Designed for ~20-30 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 1d EMA50

name = "4h_Camarilla_R3S3_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 1d data for Camarilla pivot calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)  # Previous day's close
    high_1d_shifted = np.roll(high_1d, 1)   # Previous day's high
    low_1d_shifted = np.roll(low_1d, 1)     # Previous day's low
    
    # Set first value to NaN (no previous day)
    close_1d_shifted[0] = np.nan
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    
    camarilla_r3 = close_1d_shifted + 1.1 * (high_1d_shifted - low_1d_shifted) * 1.1 / 4
    camarilla_s3 = close_1d_shifted - 1.1 * (high_1d_shifted - low_1d_shifted) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume MA and 1d EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla S3 (reversal signal)
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla R3 (reversal signal)
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above Camarilla R3, 1d EMA50 up-trend, volume confirmed
            if curr_high > curr_r3 and curr_close > curr_ema50_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3, 1d EMA50 down-trend, volume confirmed
            elif curr_low < curr_s3 and curr_close < curr_ema50_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals