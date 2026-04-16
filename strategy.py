#!/usr/bin/env python3
# 4h_KAMA_RSI_Chop - KAMA trend direction with RSI filter and Choppiness regime
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
# In trending markets (Choppiness < 38.2), follow KAMA direction with RSI confirmation.
# In ranging markets (Choppiness > 61.8), fade extremes with RSI reversal.
# Uses 4h for execution, 1d for Choppiness filter to avoid noise.
# Target: 20-50 trades over 4 years (5-12.5/year) with high-conviction entries.
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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for Choppiness filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === KAMA (adaptive moving average) ===
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.concatenate([np.array([0]), volatility[1:]])
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_4h = kama(close_4h, length=10, fast=2, slow=30)
    
    # === RSI (14-period) ===
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_4h = rsi(close_4h, length=14)
    
    # === Choppiness Index (14-period) ===
    def choppiness(high, low, close, length=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of True Range over period
        tr_sum = np.zeros_like(close)
        for i in range(length, len(close)):
            tr_sum[i] = np.nansum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            max_high[i] = np.max(high[i-length+1:i+1])
            min_low[i] = np.min(low[i-length+1:i+1])
        
        # Choppiness formula
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if tr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(length)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = choppiness(high_1d, low_1d, close_1d, length=14)
    
    # === Volume confirmation (4h) ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    # Align all HTF data to 4h timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for KAMA, RSI, and Choppiness
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_4h_aligned[i]) or 
            np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_4h_aligned[i]
        rsi_val = rsi_4h_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trending regime (Choppiness < 38.2): follow KAMA with RSI filter
            if chop_val < 38.2:
                # LONG: Price above KAMA AND RSI not overbought
                if price > kama_val and rsi_val < 70 and vol_ratio > 1.3:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Price below KAMA AND RSI not oversold
                elif price < kama_val and rsi_val > 30 and vol_ratio > 1.3:
                    signals[i] = -0.25
                    position = -1
                    continue
            # Ranging regime (Choppiness > 61.8): fade extremes with RSI reversal
            elif chop_val > 61.8:
                # LONG: RSI oversold AND price near support (below KAMA)
                if rsi_val < 30 and price < kama_val and vol_ratio > 1.2:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: RSI overbought AND price near resistance (above KAMA)
                elif rsi_val > 70 and price > kama_val and vol_ratio > 1.2:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_Chop"
timeframe = "4h"
leverage = 1.0