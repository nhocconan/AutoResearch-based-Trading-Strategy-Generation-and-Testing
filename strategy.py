#!/usr/bin/env python3
# 4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses Camarilla pivot levels (R3/S3) from daily chart with volume spike confirmation and 1-day EMA50 trend filter.
# Enters long when price breaks above R3 with volume > 1.5x 20-period average and close > EMA50.
# Enters short when price breaks below S3 with volume > 1.5x 20-period average and close < EMA50.
# Exits when price returns to the Camarilla midpoint (P) or reverses with opposite volume spike.
# Uses 1-day EMA50 to avoid whipsaws and works in both bull/bear markets.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 for breakouts
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    # Also calculate pivot point (P) for exit
    p = (prev_high + prev_low + prev_close) / 3.0
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    p_aligned = align_htf_to_ltf(prices, df_1d, p)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(p_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike in uptrend
            if (close[i] > r3_aligned[i] and 
                volume_spike[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike in downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume_spike[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot point (P) or loses trend
            if (close[i] < p_aligned[i] or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot point (P) or loses trend
            if (close[i] > p_aligned[i] or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals