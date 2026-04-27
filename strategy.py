#!/usr/bin/env python3
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
    
    # Get daily data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter (vectorized)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_14 = np.full(len(close_1d), np.nan)
    if len(tr) >= 14:
        # Wilder's smoothing
        atr_14[13] = np.mean(tr[1:14])  # First ATR
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate previous day's OHLC for Camarilla (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R3 and S3 calculation (tighter levels)
    range_hl = prev_high - prev_low
    camarilla_factor = range_hl * 1.1 / 4
    r3 = prev_close + camarilla_factor
    s3 = prev_close - camarilla_factor
    
    # Align daily indicators to 12h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 2-period volume average for spike detection (12h x 2 = 1 day)
    vol_ma = np.full(n, np.nan)
    vol_period = 2
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 2
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume (stricter to reduce trades)
        vol_filter = vol_ratio > 1.8
        
        # Volatility filter: only trade when ATR is above average (avoid low volatility whipsaws)
        vol_filter_2 = atr_14_1d_aligned[i] > np.nanmedian(atr_14_1d_aligned[max(0, i-50):i]) if i >= 50 else True
        
        if position == 0:
            # Long: Price breaks above R3 with volume and volatility
            if price > r3_aligned[i] and vol_filter and vol_filter_2:
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume and volatility
            elif price < s3_aligned[i] and vol_filter and vol_filter_2:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below S3
            if price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above R3
            if price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dATR14_Volume"
timeframe = "12h"
leverage = 1.0