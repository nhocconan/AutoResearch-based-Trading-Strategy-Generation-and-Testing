#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA50 trend filter
# Camarilla pivot levels (R3/S3) act as strong support/resistance. Breakout above R3 or below S3
# with volume confirmation and 1w EMA50 trend alignment captures institutional flow.
# Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
# Designed for 12-37 trades/year to minimize fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMA50_Trend"
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
    
    # Get 1d data for Camarilla pivots and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d data
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = pivot_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = pivot_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for volume spike filter - ONCE before loop
    vol_ma_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_20_1d * 2.0)  # Volume at least 2x average
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Close breaks above Camarilla R3 AND volume spike AND 1w EMA50 uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Close breaks below Camarilla S3 AND volume spike AND 1w EMA50 downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close breaks below Camarilla S3 (reversal) OR trend reverses
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close breaks above Camarilla R3 (reversal) OR trend reverses
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals