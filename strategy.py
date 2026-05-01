#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(34) trend filter and volume confirmation (>1.5x 20-bar MA)
# Uses 4h HTF for trend alignment and 1d HTF for session filtering (08-20 UTC). Camarilla breakouts capture momentum.
# Volume confirmation filters weak moves. Discrete sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull/bear via trend filter and session focus on active hours.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # 4h HTF data for EMA and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d HTF data for session filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(34) on 4h close
    ema_4h_34 = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_34_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_34)
    
    # Camarilla pivot levels from 4h data (using previous 4h bar's OHLC)
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    camarilla_r3 = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h)
    camarilla_s3 = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC (using DatetimeIndex hour)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 34, 50)  # Need 20 for volume MA, 34 for EMA, 50 for safety
    
    for i in range(start_idx, n):
        if np.isnan(ema_4h_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 4h EMA, and volume confirmation
            if curr_close > camarilla_r3_aligned[i] and curr_close > ema_4h_34_aligned[i] and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3, below 4h EMA, and volume confirmation
            elif curr_close < camarilla_s3_aligned[i] and curr_close < ema_4h_34_aligned[i] and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below 4h EMA
            if curr_close < camarilla_s3_aligned[i] or curr_close < ema_4h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above 4h EMA
            if curr_close > camarilla_r3_aligned[i] or curr_close > ema_4h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals