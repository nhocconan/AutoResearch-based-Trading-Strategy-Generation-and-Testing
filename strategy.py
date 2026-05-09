#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Filtered
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation, and price filter near 1d VWAP.
# Uses a higher volume threshold (2.5x average) and tighter price-VWAP proximity to reduce trades and avoid overtrading.
# Designed to generate ~20-30 trades/year on 4h to avoid fee drag while maintaining edge in bull/bear markets.
# Long when 1d trend up (close > EMA34), price breaks above R3, volume > 2.5x average, and close within 0.5% of 1d VWAP.
# Short when 1d trend down (close < EMA34), price breaks below S3, volume > 2.5x average, and close within 0.5% of 1d VWAP.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Filtered"
timeframe = "4h"
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels for each 1d bar: R3, S3
    camarilla_r3_1d = np.full_like(close_1d, np.nan)
    camarilla_s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r3_1d[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 2
            camarilla_s3_1d[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 1d VWAP for additional filter
    vwap_1d = np.full_like(close_1d, np.nan)
    cumulative_volume = np.full_like(close_1d, np.nan)
    cumulative_price_volume = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]) or np.isnan(volume_1d[i]):
            if i > 0:
                vwap_1d[i] = vwap_1d[i-1]
                cumulative_volume[i] = cumulative_volume[i-1]
                cumulative_price_volume[i] = cumulative_price_volume[i-1]
            continue
            
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        price_volume = typical_price * volume_1d[i]
        
        if i == 0:
            cumulative_volume[i] = volume_1d[i]
            cumulative_price_volume[i] = price_volume
        else:
            cumulative_volume[i] = cumulative_volume[i-1] + volume_1d[i]
            cumulative_price_volume[i] = cumulative_price_volume[i-1] + price_volume
            
        if cumulative_volume[i] != 0:
            vwap_1d[i] = cumulative_price_volume[i] / cumulative_volume[i]
        else:
            vwap_1d[i] = vwap_1d[i-1] if i > 0 else typical_price
    
    # Align 1d VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(vwap_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend and price relative to VWAP (within 0.5%)
        trend_up = close[i] > ema34_1d_aligned[i]
        vwap_diff_pct = abs((close[i] - vwap_1d_aligned[i]) / vwap_1d_aligned[i]) * 100
        near_vwap = vwap_diff_pct <= 0.5
        
        if position == 0:
            # Enter long: 1d trend up + price breaks above R3 + volume confirmation + near VWAP
            if trend_up and close[i] > camarilla_r3_1d_aligned[i] and volume_ratio[i] > 2.5 and near_vwap:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + price breaks below S3 + volume confirmation + near VWAP
            elif not trend_up and close[i] < camarilla_s3_1d_aligned[i] and volume_ratio[i] > 2.5 and near_vwap:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or price breaks below S3 or price moves away from VWAP
            if not trend_up or close[i] < camarilla_s3_1d_aligned[i] or not near_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d trend turns up or price breaks above R3 or price moves away from VWAP
            if trend_up or close[i] > camarilla_r3_1d_aligned[i] or not near_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals