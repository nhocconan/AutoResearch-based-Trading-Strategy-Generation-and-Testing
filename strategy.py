#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Camarilla R3/S3 levels from 1d pivots for breakout entries,
# 1d EMA34 for trend alignment, and volume spike (>1.8x 24-bar MA) for confirmation.
# Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.30).
# Works in both bull and bear markets via trend filter and tight entry conditions.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) for breakout
    # Based on previous 1d bar's high, low, close
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_1d_close + (prev_1d_high - prev_1d_low) * 1.1 / 4
    camarilla_s3 = prev_1d_close - (prev_1d_high - prev_1d_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.8 * 24-period average volume on 12h
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 24) + 1  # 35 (for EMA34 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h Camarilla R3/S3 breakout conditions
        breakout_r3 = curr_high > camarilla_r3_aligned[i]  # Break above 1d R3
        breakdown_s3 = curr_low < camarilla_s3_aligned[i]  # Break below 1d S3
        
        if position == 0:  # Flat - look for new entries
            # Long: 1d R3 breakout AND uptrend AND volume confirmation
            if breakout_r3 and uptrend and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: 1d S3 breakdown AND downtrend AND volume confirmation
            elif breakdown_s3 and downtrend and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 1d S3 breakdown (reversal signal)
            if curr_low < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on 1d R3 breakout (reversal signal)
            if curr_high > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals