#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
# Camarilla R3/S3 levels act as strong support/resistance derived from prior day's range
# Breakout above R3 (bullish) or below S3 (bearish) with volume confirmation
# Only trade in direction of 1d EMA34 trend to avoid counter-trend whipsaws
# Discrete position sizing (0.25) minimizes fee churn
# Targets 19-50 trades/year (75-200 total over 4 years) for 4h timeframe
# Works in bull markets via buying breakouts in uptrend and in bear markets via selling breakdowns in downtrend

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6
    #          S2 = C - (H-L)*1.1/6, S1 = C - (H-L)*1.1/4, S3 = C - (H-L)*1.1/2
    # We use R3 and S3 as entry levels
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate prior day's range for Camarilla levels
    # For today's levels, we need yesterday's HLC
    h_1d_prev = np.roll(h_1d, 1)
    l_1d_prev = np.roll(l_1d, 1)
    c_1d_prev = np.roll(c_1d, 1)
    # First bar has no prior day, set to NaN
    h_1d_prev[0] = np.nan
    l_1d_prev[0] = np.nan
    c_1d_prev[0] = np.nan
    
    # Camarilla R3 and S3 from prior day
    camarilla_r3 = c_1d_prev + (h_1d_prev - l_1d_prev) * 1.1 / 4
    camarilla_s3 = c_1d_prev - (h_1d_prev - l_1d_prev) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to complete)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and to avoid NaN from roll)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla R3 (breakout) + 1d close > EMA34 (uptrend) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S3 (breakdown) + 1d close < EMA34 (downtrend) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla S3 (reversal below S3) or 1d trend breaks
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla R3 (reversal above R3) or 1d trend breaks
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals