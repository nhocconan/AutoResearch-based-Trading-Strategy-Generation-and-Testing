#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation. Enter long when price breaks above R3 in uptrend with volume > 1.5x average, short when breaks below S3 in downtrend with volume spike. Uses discrete sizing 0.25 to limit trades (~20-40/year). Works in bull/bear via 1d trend filter and volatility expansion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate previous day's Camarilla levels (using prior 1d bar)
    # Need to shift 1d data by 1 bar to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3_1d = close_1d_prev + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3_1d = close_1d_prev - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (with proper delay for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume spike filter: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 20 for volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_ratio = volume[i] / volume_ma20[i] if volume_ma20[i] > 0 else 0
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Volume confirmation: significant volume spike
            volume_spike = vol_ratio > 1.5
            
            # Long entry: price breaks above R3 in uptrend with volume spike
            long_entry = (close_val > r3_level) and (close_val > ema_50_val) and volume_spike
            # Short entry: price breaks below S3 in downtrend with volume spike
            short_entry = (close_val < s3_level) and (close_val < ema_50_val) and volume_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price re-enters Camarilla range
            # Exit if trend turns bearish OR price moves back below R3 (failed breakout)
            if close_val < ema_50_val or close_val < r3_level:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Camarilla range
            # Exit if trend turns bullish OR price moves back above S3 (failed breakdown)
            if close_val > ema_50_val or close_val > s3_level:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0