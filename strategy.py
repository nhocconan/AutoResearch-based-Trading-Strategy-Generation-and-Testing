#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses Camarilla pivot levels (R3/S3) from daily timeframe with volume confirmation.
# Enters long when price closes above R3 with volume > 1.5x average volume.
# Enters short when price closes below S3 with volume > 1.5x average volume.
# Exits when price returns to the daily pivot point (PP).
# Uses daily trend filter (close > EMA20) to avoid counter-trend trades.
# Targets 12-37 trades per year on 12h timeframe with position size 0.25.

name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(20) for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    # R3 = PP + (HIGH - LOW) * 1.1/2
    # S3 = PP - (HIGH - LOW) * 1.1/2
    r3 = pp + (prev_high - prev_low) * 1.1 / 2.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align daily levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate average volume (20-period) for volume spike filter
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below daily EMA20
        price_above_ema = close[i] > ema_20_1d_aligned[i]
        price_below_ema = close[i] < ema_20_1d_aligned[i]
        
        # Volume spike filter: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price closes above R3 with volume spike in uptrend
            if (close[i] > r3_aligned[i] and 
                volume_spike and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price closes below S3 with volume spike in downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_spike and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to or below pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to or above pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals