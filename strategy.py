#!/usr/bin/env python3
"""
4H_Camarilla_R3S3_Breakout_1DTrend_Volume
Hypothesis: Combine Camarilla pivot levels (R3/S3) with 1-day EMA34 trend filter and volume confirmation.
In bull markets: Buy near S3/S4 with 1D uptrend and volume spike.
In bear markets: Sell near R3/R4 with 1D downtrend and volume spike.
Camarilla levels provide precise intraday support/resistance, EMA34 filters trend direction,
and volume ensures conviction. Targets 20-50 trades/year on 4H timeframe.
"""
name = "4H_Camarilla_R3S3_Breakout_1DTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels for current 4H bar using previous 1D bar
    # We'll calculate these for each 1D bar and align to 4H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    # R4 = close + (high - low) * 1.500
    # R3 = close + (high - low) * 1.250
    # S3 = close - (high - low) * 1.250
    # S4 = close - (high - low) * 1.500
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.250
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.250
    
    # Align Camarilla levels to 4H timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1D EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current 4H volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S3, 1D uptrend, and volume confirmation
            if (close[i] > camarilla_s3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R3, 1D downtrend, and volume confirmation
            elif (close[i] < camarilla_r3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 or trend reverses
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R3 or trend reverses
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals