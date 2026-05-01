#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 12-period median volume.
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 12-period median volume.
# Uses 1d Camarilla levels for institutional pivot points, 1d EMA50 for intermediate trend filter,
# and volume spike for breakout conviction. Discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year on 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Volume_v2"
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
    
    # Calculate 1d data for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    camarilla_r3 = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 4
    camarilla_s3 = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12-period volume median for volume confirmation
    vol_median_12 = pd.Series(volume).rolling(window=12, min_periods=12).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_median_12[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 12-period volume median
        if vol_median_12[i] <= 0 or np.isnan(vol_median_12[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_12[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND uptrend AND volume spike
            if curr_close > camarilla_r3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND downtrend AND volume spike
            elif curr_close < camarilla_s3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 OR trend turns down
            if curr_close < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 OR trend turns up
            if curr_close > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals