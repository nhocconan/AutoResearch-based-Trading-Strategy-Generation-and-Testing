# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakouts above/below daily Camarilla R3/S3 levels with 1d trend filter (price > SMA50) and volume confirmation. Designed for 6h timeframe to reduce trade frequency and capture medium-term moves in both bull and bear markets by aligning with daily trend.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily data for Camarilla pivot levels and SMA50 trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla R3 and S3 levels
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    
    # 1d SMA50 for trend filter
    sma50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        sma50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            sma50_1d[i] = (sma50_1d[i-1] * 49 + close_1d[i]) / 50
    
    # Daily volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align all indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for SMA50
    
    for i in range(start_idx, n):
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(sma50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h-equivalent daily volume
        # 1d = 4 x 6h bars, so scale daily volume to 6h equivalent
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 4.0  # Average 6h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > sma50_1d_aligned[i]
        is_downtrend = close[i] < sma50_1d_aligned[i]
        price_above_r3 = close[i] > r3_1d_aligned[i]
        price_below_s3 = close[i] < s3_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, in uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, in downtrend, with volume
            elif price_below_s3 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend turns down
            if not price_above_r3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S3 or trend turns up
            if not price_below_s3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals