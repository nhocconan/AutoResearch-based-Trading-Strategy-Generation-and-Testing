#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation.
# Uses Camarilla R3/S3 levels from 12h pivots for breakout entries, 12h EMA34 for trend alignment,
# and volume spike (>2.0x 20-bar MA) for confirmation. Designed for 4h timeframe to achieve
# 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25). Tighter volume
# confirmation and intermediate Camarilla levels reduce false signals while maintaining edge
# in both bull and bear markets via trend filter.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h Camarilla pivot levels (R3, S3) for breakout
    # Based on previous 12h bar's high, low, close
    prev_12h_high = df_12h['high'].shift(1).values
    prev_12h_low = df_12h['low'].shift(1).values
    prev_12h_close = df_12h['close'].shift(1).values
    
    camarilla_r3 = prev_12h_close + (prev_12h_high - prev_12h_low) * 1.1 / 4
    camarilla_s3 = prev_12h_close - (prev_12h_high - prev_12h_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20) + 1  # 51 (for EMA34 and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA34 direction
        uptrend = curr_close > ema_34_12h_aligned[i]
        downtrend = curr_close < ema_34_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h Camarilla R3/S3 breakout conditions
        breakout_r3 = curr_high > camarilla_r3_aligned[i]  # Break above 12h R3
        breakdown_s3 = curr_low < camarilla_s3_aligned[i]  # Break below 12h S3
        
        if position == 0:  # Flat - look for new entries
            # Long: 12h R3 breakout AND uptrend AND volume confirmation
            if breakout_r3 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 12h S3 breakdown AND downtrend AND volume confirmation
            elif breakdown_s3 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 12h S3 breakdown (reversal signal)
            if curr_low < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on 12h R3 breakout (reversal signal)
            if curr_high > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals