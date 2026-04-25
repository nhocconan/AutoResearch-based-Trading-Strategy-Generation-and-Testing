#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above R3 in 1-week uptrend (close > 1w EMA50) with volume > 2.0x 20-period average.
Short when price breaks below S3 in 1-week downtrend (close < 1w EMA50) with volume > 2.0x 20-period average.
Exit via opposite Camarilla level (S3 for longs, R3 for shorts).
Designed for ~7-25 trades/year via tight R3/S3 breakout conditions on daily timeframe.
Uses 1-week trend filter to work in both bull and bear markets, avoiding false breakouts via volume confirmation.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    # Get daily OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3/S3)
    # R3 = c + (h-l)*1.1/4
    # S3 = c - (h-l)*1.1/4
    camarilla_r3_1d = c_1d + ((h_1d - l_1d) * 1.1 / 4)
    camarilla_s3_1d = c_1d - ((h_1d - l_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (no alignment needed as both are 1d)
    camarilla_r3_aligned = camarilla_r3_1d
    camarilla_s3_aligned = camarilla_s3_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: break above R3 with volume confirmation
                long_signal = (close[i] > r3_level) and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: break below S3 with volume confirmation
                short_signal = (close[i] < s3_level) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Price breaks below S3 (opposite Camarilla level)
            if close[i] < s3_level:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Price breaks above R3 (opposite Camarilla level)
            if close[i] > r3_level:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0