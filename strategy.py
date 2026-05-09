#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Reversal_Volume
Hypothesis: In 4h timeframe, use Kaufman Adaptive Moving Average (KAMA) to determine trend direction.
Enter long when price crosses above KAMA, RSI < 30 (oversold), and volume > 1.5x average.
Enter short when price crosses below KAMA, RSI > 70 (overbought), and volume > 1.5x average.
Use daily timeframe for volatility filter: only trade when daily ATR(14) is above its 50-period MA (high volatility regime).
Exit when price crosses back across KAMA or after 8 bars to limit exposure.
Designed for low trade frequency (~20-40/year) with discipline in both bull and bear markets.
"""

name = "4h_KAMA_Trend_RSI_Reversal_Volume"
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
    
    # Get daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index
    
    atr_14_1d = np.full_like(tr_1d, np.nan)
    if len(tr_1d) >= 14:
        atr_14_1d[13] = np.nanmean(tr_1d[1:15])  # first 14 values
        for i in range(15, len(tr_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Daily ATR(14) 50-period MA for volatility regime filter
    atr_ma_50_1d = np.full_like(atr_14_1d, np.nan)
    if len(atr_14_1d) >= 50:
        valid_50 = ~np.isnan(atr_14_1d)
        if np.sum(valid_50) >= 50:
            # Simple approach: wait for 50 valid values
            atr_values = atr_14_1d[valid_50]
            if len(atr_values) >= 50:
                ma_50 = np.full_like(atr_values, np.nan)
                ma_50[49] = np.mean(atr_values[0:50])
                for i in range(50, len(atr_values)):
                    ma_50[i] = (ma_50[i-1] * 49 + atr_values[i]) / 50
                # Map back to original array
                atr_ma_50_1d[valid_50] = ma_50
    
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Calculate KAMA on 4h data
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[:length] = np.nan
        for i in range(length, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[:length] = np.nan
        kama_vals[length] = close[length]  # seed
        for i in range(length+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, length=10, fast=2, slow=30)
    
    # RSI(14) on 4h
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.zeros_like(close)
        rs[:length+1] = np.nan
        for i in range(length+1, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
            else:
                rs[i] = 0
        
        rsi_vals = np.zeros_like(close)
        rsi_vals[:length+1] = np.nan
        for i in range(length+1, len(close)):
            rsi_vals[i] = 100 - (100 / (1 + rs[i]))
        return rsi_vals
    
    rsi_vals = rsi(close, length=14)
    
    # Volume ratio: current / 20-period average
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
    
    start_idx = max(20, 30)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr_ma_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Volatility filter: only trade when daily ATR(14) > its 50 MA
        volatility_filter = atr_14_1d[i//16] > atr_ma_50_1d[i//16] if i//16 < len(atr_14_1d) else False
        
        bars_since_entry += 1
        
        if position == 0:
            if volatility_filter:
                # Enter long: price > KAMA, RSI < 30 (oversold), volume spike
                if (close[i] > kama_vals[i] and 
                    rsi_vals[i] < 30 and 
                    volume_ratio[i] > 1.5):
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Enter short: price < KAMA, RSI > 70 (overbought), volume spike
                elif (close[i] < kama_vals[i] and 
                      rsi_vals[i] > 70 and 
                      volume_ratio[i] > 1.5):
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        
        elif position == 1:
            # Exit conditions: price < KAMA OR max 8 bars
            if close[i] < kama_vals[i] or bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price > KAMA OR max 8 bars
            if close[i] > kama_vals[i] or bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals