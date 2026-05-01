#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA34 as trend filter (stronger HTF trend) and 1d Camarilla R3/S3 for breakout signals.
# Volume confirmation requires current 1d volume > 2.0 * 20-period average volume.
# Only takes longs in uptrend (price > 1w EMA34) on R3 breakout, shorts in downtrend on S3 breakdown.
# Discrete position sizing 0.25 to balance return and drawdown. Target: 30-100 trades over 4 years.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) - stronger breakout levels
    # Camarilla: based on previous day's high, low, close
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20) + 1  # 35 (for EMA34 and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA34 direction
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Camarilla R3/S3 breakout conditions
        breakout_r3 = curr_close > camarilla_r3[i]  # Break above R3
        breakdown_s3 = curr_close < camarilla_s3[i]  # Break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 AND uptrend AND volume confirmation
            if breakout_r3 and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 AND downtrend AND volume confirmation
            elif breakdown_s3 and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S3 (reversal signal)
            if curr_close < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R3 (reversal signal)
            if curr_close > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals