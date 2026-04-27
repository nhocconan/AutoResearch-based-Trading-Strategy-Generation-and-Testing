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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for trend filter (vectorized with proper initialization)
    close_1d = df_1d['close'].values
    rsi_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        # Calculate RSI using Wilder's smoothing
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Initial average gain and loss
        avg_gain = np.mean(gain[:14])
        avg_loss = np.mean(loss[:14])
        
        if avg_loss == 0:
            rsi_14_1d[13] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_14_1d[13] = 100 - (100 / (1 + rs))
        
        # Calculate RSI for remaining values
        for i in range(14, len(close_1d)):
            if avg_loss == 0:
                rsi_14_1d[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi_14_1d[i] = 100 - (100 / (1 + rs))
            
            # Update smoothed averages
            avg_gain = (avg_gain * 13 + gain[i-1]) / 14
            avg_loss = (avg_loss * 13 + loss[i-1]) / 14
    
    # Calculate previous day's OHLC for Camarilla (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R3 and S3 calculation (tighter bands for fewer trades)
    range_hl = prev_high - prev_low
    camarilla_factor = range_hl * 1.1 / 6
    r3 = prev_close + camarilla_factor
    s3 = prev_close - camarilla_factor
    
    # Align daily indicators to 1h timeframe
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above R3 with volume and RSI > 50 (bullish bias)
            if price > r3_aligned[i] and vol_filter and rsi_14_1d_aligned[i] > 50:
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume and RSI < 50 (bearish bias)
            elif price < s3_aligned[i] and vol_filter and rsi_14_1d_aligned[i] < 50:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below S3 or RSI < 40
            if price < s3_aligned[i] or rsi_14_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above R3 or RSI > 60
            if price > r3_aligned[i] or rsi_14_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_1dRSI14_Volume"
timeframe = "1h"
leverage = 1.0