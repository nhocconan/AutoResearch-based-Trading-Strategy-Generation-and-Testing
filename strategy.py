#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) from weekly chart act as strong support/resistance - breaks often lead to sustained moves
# 1w EMA50 ensures alignment with higher-timeframe trend to avoid counter-trend trades in both bull/bear markets
# Volume spike (>1.8 x 30-period EMA) confirms breakout validity and reduces false signals
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns
# Works in bull markets by catching breakouts with trend, works in bear by only taking trend-aligned breaks

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.8 x 30-period EMA)
    vol_ema_30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_30)
    
    # 1w data for Camarilla pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1w bar
    # Based on previous week's high, low, close
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Camarilla levels: R3/S3 are the most significant for breakouts
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    camarilla_r3 = pivot + (range_ * 1.1 / 4.0)  # R3 level
    camarilla_s3 = pivot - (range_ * 1.1 / 4.0)  # S3 level
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Camarilla R3 with volume confirmation and uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 with volume confirmation and downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Camarilla S3 (reversal to downside) OR trend changes to downtrend
            if close[i] < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Camarilla R3 (reversal to upside) OR trend changes to uptrend
            if close[i] > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals