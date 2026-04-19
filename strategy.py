#!/usr/bin/env python3
"""
4h_4h_KAMA_1d_CSI30_RSI_Trend_Follow
Hypothesis: KAMA on 4h identifies trend direction, RSI on 1d filters overbought/oversold extremes,
combined with CSI30 on 1d for regime filtering. Works in bull/bear via trend alignment and
extreme RSI filtering to avoid counter-trend trades in strong trends.
"""

name = "4h_4h_KAMA_1d_CSI30_RSI_Trend_Follow"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA on 4h for trend direction (fast=2, slow=30)
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, length))
        change[0:length] = np.nansum(np.abs(np.diff(close[:length+1]))) if length > 0 else 0
        
        vol = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        vol_arr = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i == 0:
                vol_arr[i] = 0
            else:
                vol_arr[i] = vol_arr[i-1] + np.abs(close[i] - close[i-1])
        
        er = np.where(vol_arr != 0, change / vol_arr, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        
        # KAMA calculation
        kama_out = np.full_like(close, np.nan)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama_out[i-1]):
                kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
            else:
                kama_out[i] = close[i]
        return kama_out
    
    # CSI30 (Choppiness Index) on 1d for regime filtering
    def csi(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.full_like(close, np.nan)
        for i in range(length, len(tr)):
            atr[i] = np.nanmean(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.full_like(close, np.nan)
        ll = np.full_like(close, np.nan)
        for i in range(length-1, len(close)):
            hh[i] = np.nanmax(high[i-length+1:i+1])
            ll[i] = np.nanmin(low[i-length+1:i+1])
        
        # CSI calculation
        csi_out = np.full_like(close, np.nan)
        for i in range(length-1, len(close)):
            if not np.isnan(atr[i]) and atr[i] > 0:
                csi_out[i] = 100 * np.log10((hh[i] - ll[i]) / (atr[i] * length)) / np.log10(length)
        return csi_out
    
    # RSI on 1d for overbought/oversold filtering
    def rsi(close, length=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        for i in range(length, len(close)):
            if i == length:
                avg_gain[i] = np.nanmean(gain[i-length+1:i+1])
                avg_loss[i] = np.nanmean(loss[i-length+1:i+1])
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    # Get 4h data for KAMA (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 4h
    kama_4h = kama(df_4h['close'].values, length=10, fast=2, slow=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Get 1d data for CSI30 and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate CSI30 on 1d
    csi_1d = csi(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, length=14)
    csi_1d_aligned = align_htf_to_ltf(prices, df_1d, csi_1d)
    
    # Calculate RSI on 1d
    rsi_1d = rsi(df_1d['close'].values, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(csi_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from KAMA: price > KAMA = uptrend, price < KAMA = downtrend
        kama_trend = 1 if close[i] > kama_4h_aligned[i] else -1
        
        # CSI30 filter: CSI > 50 = ranging/choppy, CSI < 50 = trending
        # We prefer trending markets (CSI < 50) for trend following
        trending_regime = csi_1d_aligned[i] < 50
        
        # RSI filter: avoid extreme overbought/oversold for trend following
        rsi_not_extreme = (rsi_1d_aligned[i] > 20) and (rsi_1d_aligned[i] < 80)
        
        if position == 0:
            # Long: price above KAMA (uptrend), trending regime, not extreme RSI, volume confirmation
            if (kama_trend == 1 and 
                trending_regime and 
                rsi_not_extreme and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), trending regime, not extreme RSI, volume confirmation
            elif (kama_trend == -1 and 
                  trending_regime and 
                  rsi_not_extreme and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or regime becomes choppy
            if (kama_trend == -1) or (not trending_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or regime becomes choppy
            if (kama_trend == 1) or (not trending_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals