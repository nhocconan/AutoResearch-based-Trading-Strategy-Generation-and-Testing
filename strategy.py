#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3 level, 1w EMA50 uptrend, and volume > 2.0x 20-bar average.
# Enter short when price breaks below Camarilla S3 level, 1w EMA50 downtrend, and volume > 2.0x 20-bar average.
# Exit when price reaches opposite Camarilla level (R3->S3 or S3->R3) or volume drops below average.
# Uses discrete position sizing (0.30) to balance return and fee drag.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.
# Camarilla levels provide precise intraday pivot points; 1w trend filter ensures alignment with higher timeframe momentum.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    true_range = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r3 = close_1d + (true_range * 1.1 / 4)
    camarilla_s3 = close_1d - (true_range * 1.1 / 4)
    camarilla_r4 = close_1d + (true_range * 1.1 / 2)
    camarilla_s4 = close_1d - (true_range * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1w EMA50 slope
        if i >= 1:
            ema_prev = ema_50_1w_aligned[i-1]
            ema_curr = ema_50_1w_aligned[i]
            trend_up = ema_curr > ema_prev
            trend_down = ema_curr < ema_prev
        else:
            trend_up = True
            trend_down = True
        
        # Camarilla breakout conditions
        breakout_r3 = close[i] > camarilla_r3_aligned[i-1]  # Break above R3
        breakout_s3 = close[i] < camarilla_s3_aligned[i-1]  # Break below S3
        
        # Exit conditions: reach opposite Camarilla level or volume drop
        exit_long = close[i] < camarilla_s3_aligned[i] or not vol_confirm
        exit_short = close[i] > camarilla_r3_aligned[i] or not vol_confirm
        
        # Handle entries and exits
        if breakout_r3 and trend_up and vol_confirm and position <= 0:
            signals[i] = 0.30
            position = 1
        elif breakout_s3 and trend_down and vol_confirm and position >= 0:
            signals[i] = -0.30
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals