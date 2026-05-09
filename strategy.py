#!/usr/bin/env python3
# 4h_KAMA_RSI_Chop_Filter - KAMA direction + RSI + chop filter for 4h timeframe
# Hypothesis: KAMA adapts to trend direction, RSI filters overbought/oversold conditions,
# and Choppiness Index identifies ranging vs trending markets. This combination should
# work in both bull and bear markets by avoiding whipsaws in ranging conditions and
# capturing trends when they emerge. Designed for low trade frequency to minimize fee drag.

name = "4h_KAMA_RSI_Chop_Filter"
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
    
    # KAMA calculation (adaptive moving average)
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([0])
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / np.maximum(volatility[length-1:], 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama_vals = np.full_like(close, np.nan)
        kama_vals[length] = close[length]
        for i in range(length+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(close) >= length+1:
            avg_gain[length] = np.mean(gain[:length])
            avg_loss[length] = np.mean(loss[:length])
            for i in range(length+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Choppiness Index calculation
    def choppiness_index(high, low, close, length=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.full_like(close, np.nan)
        if len(close) >= length+1:
            atr[length] = np.nanmean(tr[1:length+1])
            for i in range(length+1, len(close)):
                atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        # Highest high and lowest low over period
        highest_high = np.full_like(close, np.nan)
        lowest_low = np.full_like(close, np.nan)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        # Chop calculation
        chop = np.full_like(close, 50.0)
        for i in range(length, len(close)):
            if not np.isnan(atr[i]) and atr[i] > 0:
                sum_tr = np.nansum(tr[i-length+1:i+1])
                hh_ll = highest_high[i] - lowest_low[i]
                if hh_ll > 0:
                    chop[i] = 100 * np.log10(sum_tr / hh_ll) / np.log10(length)
        return chop
    
    # Calculate indicators
    kama_vals = kama(close, length=10, fast=2, slow=30)
    rsi_vals = rsi(close, length=14)
    chop_vals = choppiness_index(high, low, close, length=14)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(30, 20, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price above KAMA (uptrend), RSI not overbought, chop < 61.8 (trending), volume spike
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] < 70 and 
                chop_vals[i] < 61.8 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price below KAMA (downtrend), RSI not oversold, chop < 61.8 (trending), volume spike
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] > 30 and 
                  chop_vals[i] < 61.8 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI overbought OR chop > 61.8 (ranging)
            if (close[i] < kama_vals[i] or 
                rsi_vals[i] > 70 or 
                chop_vals[i] > 61.8):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI oversold OR chop > 61.8 (ranging)
            if (close[i] > kama_vals[i] or 
                rsi_vals[i] < 30 or 
                chop_vals[i] > 61.8):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals