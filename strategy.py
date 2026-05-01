#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses proven Camarilla structure from top performers: tight R3/S3 levels for high-quality breakouts.
# Trend filter ensures trades align with higher-timeframe direction. Volume confirmation filters false breakouts.
# Works in bull (buy breakouts with uptrend) and bear (sell breakdowns with downtrend).
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Discrete position sizing (0.30) to balance return and fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Load 1d OHLC for Camarilla calculation (once before loop)
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    prev_close = df_1d_ohlc['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (proven breakout levels from top performers)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.0 / 4.0
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.0 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA34 and volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Camarilla breakout conditions (R3/S3 for continuation)
        breakout_up = curr_close > camarilla_r3_aligned[i]   # break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i] # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume confirmation
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Breakout down AND downtrend AND volume confirmation
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakout down (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout up (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals