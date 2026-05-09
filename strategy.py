#!/usr/bin/env python3
name = "6H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla R3/S3 levels from previous day (more extreme levels)
    # R3 = C + 1.1*(H-L)*1.1/4, S3 = C - 1.1*(H-L)*1.1/4
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_r3[i] = prev_close + 1.1 * range_ * 1.1 / 4
        camarilla_s3[i] = prev_close - 1.1 * range_ * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for trend filter (EMA34)
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(34, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        # Downtrend: price below 1d EMA34
        downtrend = close[i] < ema34_1d_aligned[i]
        # Volume spike: current volume > 2.5x average volume (using 6-period EMA of volume)
        if i >= 6:
            vol_ema6 = pd.Series(volume[:i+1]).ewm(span=6, adjust=False).mean().iloc[-1]
            volume_spike = volume[i] > vol_ema6 * 2.5
        else:
            volume_spike = False
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Camarilla R3 + volume spike
            if uptrend and close[i] > camarilla_r3_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Camarilla S3 + volume spike
            elif downtrend and close[i] < camarilla_s3_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Camarilla S3 (mean reversion)
            if not uptrend or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Camarilla R3 (mean reversion)
            if not downtrend or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals