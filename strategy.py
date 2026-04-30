#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation (>2.0x average)
# Camarilla R3/S3 levels act as intraday support/resistance; breakouts with volume and higher timeframe trend
# filter capture momentum moves while reducing false signals. Works in bull/bear via EMA34 trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily pivot points for Camarilla levels
    df_1d_for_pivot = df_1d.copy()
    df_1d_for_pivot['high'] = df_1d['high'].values
    df_1d_for_pivot['low'] = df_1d['low'].values
    df_1d_for_pivot['close'] = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    prev_close = df_1d_for_pivot['close'].shift(1).values
    prev_high = df_1d_for_pivot['high'].shift(1).values
    prev_low = df_1d_for_pivot['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla R3, R4, S3, S4 levels
    r3 = pivot + (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1)
    
    # AlCamarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 24-period average (4 days of 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)  # warmup for EMA34 (34)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Camarilla breakout
            if curr_volume_spike:
                # Bullish: price breaks above Camarilla R3 + price above 1d EMA34
                if curr_high > r3_aligned[i-1] and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Camarilla S3 + price below 1d EMA34
                elif curr_low < s3_aligned[i-1] and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 (trend reversal)
            if curr_low < s3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 (trend reversal)
            if curr_high > r3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals