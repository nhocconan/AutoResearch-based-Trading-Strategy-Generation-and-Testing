#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R4/S4 breakouts on 6h with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above R4 in uptrend (close > 1d EMA50) with volume spike.
Short when price breaks below S4 in downtrend (close < 1d EMA50) with volume spike.
Exit when price retraces to R3/S3 or trend reverses. Targets 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for each 6h bar using prior 6h bar's range
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    # R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low)
    range_6h = high_6h - low_6h
    r4_6h = close_6h + 1.5 * range_6h
    s4_6h = close_6h - 1.5 * range_6h
    r3_6h = close_6h + 1.125 * range_6h
    s3_6h = close_6h - 1.125 * range_6h
    
    # Align Camarilla levels to original timeframe
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (1d)
                # Long: break above R4 with volume spike
                long_signal = (close[i] > r4_aligned[i]) and vol_spike[i]
                # Short: break below S4 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < s4_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (1d)
                # Short: break below S4 with volume spike
                short_signal = (close[i] < s4_aligned[i]) and vol_spike[i]
                # Long: break above R4 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > r4_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: retrace to R3 or trend reversal
            exit_signal = (close[i] < r3_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: retrace to S3 or trend reversal
            exit_signal = (close[i] > s3_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0