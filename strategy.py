#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R4/S4) from 1w for stronger institutional support/resistance
# 1w EMA50 ensures alignment with weekly trend for high-probability entries
# Volume spike (2.5x 20-period average) confirms strong institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) for 4h timeframe
# Works in bull markets via R4 breakout continuation and in bear markets via S4 breakdown continuation

name = "4h_Camarilla_R4_S4_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Camarilla levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1w Camarilla levels (R4, S4) - stronger levels than R3/S3
    # Camarilla: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    weekly_range = df_1w['high'].values - df_1w['low'].values
    camarilla_r4 = df_1w['close'].values + 1.1 * weekly_range
    camarilla_s4 = df_1w['close'].values - 1.1 * weekly_range
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Calculate volume spike (2.5x 20-period average) - stricter than 2.0x
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and Camarilla)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R4 + 1w close > EMA50 + volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4 + 1w close < EMA50 + volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < ema_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S4 or 1w trend breaks
            if close[i] < camarilla_s4_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R4 or 1w trend breaks
            if close[i] > camarilla_r4_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals