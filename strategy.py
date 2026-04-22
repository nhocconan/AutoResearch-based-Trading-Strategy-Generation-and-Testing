#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1-day data for pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day True Range for volatility
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h ATR for volatility and entry conditions
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h SMA for trend context
    sma = pd.Series(close).rolling(window=30, min_periods=30).mean().values
    
    # 6h volume moving average for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily pivot points (classic)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot_1d - low_1d
    s1 = 2 * pivot_1d - high_1d
    r2 = pivot_1d + (high_1d - low_1d)
    s2 = pivot_1d - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot_1d - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align daily pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Align daily ATR for volatility regime filter
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(sma[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_val = atr[i]
        sma_val = sma[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        atr_1d = atr_1d_aligned[i]
        
        # Get current pivot levels
        pivot = pivot_1d_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        
        # Volatility regime: only trade when volatility is elevated (trending market)
        vol_regime = atr_1d > np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        
        # Volume confirmation: above average volume
        vol_confirm = vol > vol_ma_val * 1.2
        
        if position == 0 and vol_regime and vol_confirm:
            # Long conditions: price above S1 and breaking above R1 with momentum
            if price > s1_level and price > r1_level and price > sma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short conditions: price below R1 and breaking below S1 with momentum
            elif price < r1_level and price < s1_level and price < sma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions
            if position == 1:  # Long position
                # Exit if price breaks below S1 (support) or reaches R3 (strong resistance)
                if price < s1_level or price > r3_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                # Exit if price breaks above R1 (resistance) or reaches S3 (strong support)
                if price > r1_level or price < s3_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals

name = "6h_PivotPoint_R1_S1_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0