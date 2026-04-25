#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 in uptrend (close > 1d EMA50) with volume > 2x 20-period average.
Short when price breaks below S3 in downtrend (close < 1d EMA50) with volume > 2x 20-period average.
Exit when price reverts to Camarilla H3/L3 levels or trend reverses. Designed for low trade frequency (<30/year) and robustness in both bull and bear markets.
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
    
    # Get 12h data for Camarilla calculations (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_r3 = np.zeros(len(df_12h))
    camarilla_s3 = np.zeros(len(df_12h))
    camarilla_h3 = np.zeros(len(df_12h))
    camarilla_l3 = np.zeros(len(df_12h))
    
    for i in range(len(df_12h)):
        if i < 1:  # Need previous bar for calculation
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            continue
            
        # Use previous bar's high, low, close
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        
        rng = ph - pl
        camarilla_r3[i] = pc + (rng * 1.1 / 4)
        camarilla_s3[i] = pc - (rng * 1.1 / 4)
        camarilla_h3[i] = pc + (rng * 1.1 / 6)
        camarilla_l3[i] = pc - (rng * 1.1 / 6)
    
    # Align Camarilla levels to original timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
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
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (daily)
                # Long: break above R3 with volume spike
                long_signal = (close[i] > r3_aligned[i]) and vol_spike[i]
                # Short: break below S3 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < s3_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (daily)
                # Short: break below S3 with volume spike
                short_signal = (close[i] < s3_aligned[i]) and vol_spike[i]
                # Long: break above R3 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > r3_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
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
            # Exit conditions: revert to H3 or trend reversal
            exit_signal = (close[i] < h3_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: revert to L3 or trend reversal
            exit_signal = (close[i] > l3_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0