#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and 1d EMA34 trend filter.
# Camarilla levels (R3/S3) act as strong support/resistance; reversal on touch with volume confirms institutional interest.
# EMA34 on 1d filters for trend alignment. Works in both bull (buy S3 in uptrend) and bear (sell R3 in downtrend).
# Low-frequency signals reduce fee drag; pivot levels provide structure in ranging markets.
name = "4h_Camarilla_R3S3_Reversal_1dEMA34_Volume"
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
    
    # 1d data for Camarilla pivots, trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Camarilla levels from previous 1d bar
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 1d bar
    camarilla_r3 = np.zeros_like(c_close)
    camarilla_s3 = np.zeros_like(c_close)
    for i in range(len(c_close)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            # Previous day's range
            prev_range = c_high[i-1] - c_low[i-1]
            camarilla_r3[i] = c_close[i-1] + (1.1 * prev_range / 6)
            camarilla_s3[i] = c_close[i-1] - (1.1 * prev_range / 6)
    
    # Align Camarilla levels to 4h (previous day's levels available at 4h open)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 trend filter
    ema_1d = pd.Series(c_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d volume confirmation: volume > 1.5x 20 EMA volume
    vol_ema_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_1d)
    vol_spike = df_1d['volume'].values > (1.5 * vol_ema_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price touches S3 + volume spike + price > EMA34 (uptrend)
            if (price <= camarilla_s3_aligned[i] * 1.001 and  # Allow small tolerance for touch
                vol_spike_aligned[i] and 
                price > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 + volume spike + price < EMA34 (downtrend)
            elif (price >= camarilla_r3_aligned[i] * 0.999 and  # Allow small tolerance for touch
                  vol_spike_aligned[i] and 
                  price < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above R3 (breakout) or crosses below S3 (failed hold)
            if price >= camarilla_r3_aligned[i] * 0.999 or price <= camarilla_s3_aligned[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below S3 (breakdown) or crosses above R3 (failed hold)
            if price <= camarilla_s3_aligned[i] * 1.001 or price >= camarilla_r3_aligned[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals