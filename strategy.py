#!/usr/bin/env python3
"""
6h Camarilla R3S3 Breakout + 1d ADX Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels on 1d act as significant support/resistance. 
Break above R3 with volume and 1d ADX>25 (strong trend) signals bullish momentum.
Break below S3 with volume and 1d ADX>25 signals bearish momentum.
Uses 6h timeframe for lower trade frequency. Works in bull/bear via ADX trend filter.
Volume spike confirms institutional participation. Target: 12-37 trades/year.
"""

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
    
    # Get 1d data for Camarilla pivot calculation and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot and ADX
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use the previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels using previous day's data
    range_hl = prev_high - prev_low
    camarilla_r3 = prev_close + (range_hl * 1.1 / 4)
    camarilla_s3 = prev_close - (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed as they're based on prev day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d ADX for trend filter (uses 14-period)
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=14).mean().values
        
        # Directional Movement
        dm_plus = pd.Series(df_1d['high']).diff()
        dm_minus = -pd.Series(df_1d['low']).diff()
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Smoothed DM and TR
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_smooth = pd.Series(atr_14).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr_smooth != 0, atr_smooth, 1)
        di_minus = 100 * dm_minus_smooth / np.where(atr_smooth != 0, atr_smooth, 1)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, 20.0)  # default to weak trend if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        adx_value = adx_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_value > 25
        
        if position == 0:
            # Long: price breaks above R3 AND volume spike AND strong trend
            long_condition = (curr_close > r3_level) and volume_spike and strong_trend
            # Short: price breaks below S3 AND volume spike AND strong trend
            short_condition = (curr_close < s3_level) and volume_spike and strong_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below S3 or trend weakens
            if curr_close <= s3_level or adx_value < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above R3 or trend weakens
            if curr_close >= r3_level or adx_value < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0