#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide institutional support/resistance with high probability reactions
# 12h EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity
# Works in bull markets (break above R1 + 12h EMA50 up) and bear markets (break below S1 + 12h EMA50 down)
# Uses discrete position sizing (0.30) to minimize fee churn and control drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    # 12h data for Camarilla pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    
    camarilla_r1 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12
    camarilla_s1 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed 12h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above R1 with volume confirmation and uptrend
            if close[i] > camarilla_r1_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S1 with volume confirmation and downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below S1 (reversal) OR trend changes to downtrend
            if close[i] < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Close rises above R1 (reversal) OR trend changes to uptrend
            if close[i] > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals