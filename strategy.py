#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_Filter_v1
Hypothesis: Trade Camarilla pivot (R3/S3) breakouts on 6h with 1d EMA50 trend filter and volume confirmation.
Only trade in direction of 1d EMA50 trend to reduce whipsaws in bear markets. Uses volume spike (2.0x) for confirmation.
Designed for 12-37 trades/year on 6h timeframe. Works in bull/bear by following 1d EMA50 trend and filtering counter-trend breaks.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close_1d + 1.500 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 1.500 * (prev_high_1d - prev_low_1d)
    camarilla_r4 = prev_close_1d + 1.625 * (prev_high_1d - prev_low_1d)
    camarilla_s4 = prev_close_1d - 1.625 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (50), volume median (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend (close > 1d EMA50)
            long_signal = (close_val > camarilla_r3_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_1d_val)
            
            # Short: break below S3 with volume spike, and downtrend (close < 1d EMA50)
            short_signal = (close_val < camarilla_s3_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            # 1. Price breaks below S3 (reversal)
            # 2. Strong break below S4 (accelerated reversal)
            # 3. Trend changes (close < 1d EMA50)
            if (close_val < camarilla_s3_aligned[i]) or \
               (close_val < camarilla_s4_aligned[i]) or \
               (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            # 1. Price breaks above R3 (reversal)
            # 2. Strong break above R4 (accelerated reversal)
            # 3. Trend changes (close > 1d EMA50)
            if (close_val > camarilla_r3_aligned[i]) or \
               (close_val > camarilla_r4_aligned[i]) or \
               (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0