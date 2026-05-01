#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Uses Camarilla R3/S3 levels from 4h pivots for breakout entries (balanced bands for fewer false signals),
# 4h EMA34 for trend alignment, and volume spike (>1.6x 20-bar MA) for confirmation.
# Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC) and discrete sizing (0.20).
# Works in both bull and bear markets via trend filter and session-based noise reduction.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to reduce noise trades
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla pivot levels (R3, S3) for breakout
    # Based on previous 4h bar's high, low, close
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    prev_4h_close = df_4h['close'].shift(1).values
    
    camarilla_r3 = prev_4h_close + (prev_4h_high - prev_4h_low) * 1.1 / 4
    camarilla_s3 = prev_4h_close - (prev_4h_high - prev_4h_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume confirmation: current volume > 1.6 * 20-period average volume on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20) + 1  # 35 (for EMA34 and volume MA)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if (np.isnan(ema_34_4h_aligned[i]) or 
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
        
        # Trend filter: 4h EMA34 direction
        uptrend = curr_close > ema_34_4h_aligned[i]
        downtrend = curr_close < ema_34_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1h Camarilla R3/S3 breakout conditions
        breakout_r3 = curr_high > camarilla_r3_aligned[i]  # Break above 4h R3
        breakdown_s3 = curr_low < camarilla_s3_aligned[i]  # Break below 4h S3
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h R3 breakout AND uptrend AND volume confirmation AND session
            if breakout_r3 and uptrend and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h S3 breakdown AND downtrend AND volume confirmation AND session
            elif breakdown_s3 and downtrend and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 4h S3 breakdown (reversal signal)
            if curr_low < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on 4h R3 breakout (reversal signal)
            if curr_high > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals