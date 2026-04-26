#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike_v2
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h with 1d ATR-based trend filter and volume confirmation (2.0x median). Trend: price > 1d close + 0.5*ATR (uptrend) or < 1d close - 0.5*ATR (downtrend). Only trade in trend direction to reduce whipsaws. Uses ATR trailing stop (1.5x ATR). Target: 20-30 trades/year on 4h. Works in bull/bear by adapting to trend and volatility.
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
    
    # 1d ATR(14) for trend filter
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close_1d + 3.000/6 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 3.000/6 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d close for trend reference
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Volume confirmation: 2.0x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops (4h ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of ATR(14) 1d, volume median (20), ATR (14) 4h
    start_idx = max(14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(prev_close_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        atr_1d_val = atr_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        prev_close_1d_val = prev_close_1d_aligned[i]
        
        # Trend filter: price > 1d close + 0.5*ATR1d (uptrend) or < 1d close - 0.5*ATR1d (downtrend)
        uptrend = close_val > prev_close_1d_val + 0.5 * atr_1d_val
        downtrend = close_val < prev_close_1d_val - 0.5 * atr_1d_val
        
        if position == 0:
            # Long: break above R3 with volume spike, and uptrend
            long_signal = (close_val > camarilla_r3_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          uptrend
            
            # Short: break below S3 with volume spike, and downtrend
            short_signal = (close_val < camarilla_s3_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: 1.5x ATR
            if close_val < highest_since_entry - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: 1.5x ATR
            if close_val > lowest_since_entry + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dATR_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0