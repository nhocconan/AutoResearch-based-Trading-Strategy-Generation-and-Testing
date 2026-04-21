#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Hypothesis: Breakouts of Donchian channels (20-period) on 4h timeframe with 
volume confirmation and 4h trend filter (HMA 21). Trend filter prevents 
counter-trend trades. Volume ensures breakout strength. Works in bull markets 
by capturing breakouts and in bear markets by capturing breakdowns. 
Target: 20-40 trades/year per symbol, low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    # WMA function
    def wma(x, w):
        if len(x) < w:
            return np.full_like(x, np.nan)
        weights = np.arange(1, w + 1)
        return np.convolve(x, weights, 'valid') / weights.sum()
    
    wma_half = wma(arr, half)
    wma_full = wma(arr, period)
    if len(wma_half) == 0 or len(wma_full) == 0:
        return np.full(n, np.nan)
    
    raw = 2 * wma_half - wma_full
    hma = wma(raw, sqrt)
    
    # Pad with NaN to match original length
    result = np.full(n, np.nan)
    start_idx = period - half  # rough approximation
    if start_idx < len(hma):
        end_idx = start_idx + len(hma)
        if end_idx <= n:
            result[start_idx:end_idx] = hma[:n-start_idx]
        else:
            result[start_idx:n] = hma[:n-start_idx]
    return result

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian, HMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels (20-period)
    if len(high_4h) >= 20:
        dc_high = np.full(len(high_4h), np.nan)
        dc_low = np.full(len(high_4h), np.nan)
        for i in range(19, len(high_4h)):
            dc_high[i] = np.max(high_4h[i-19:i+1])
            dc_low[i] = np.min(low_4h[i-19:i+1])
    else:
        dc_high = np.full(len(high_4h), np.nan)
        dc_low = np.full(len(high_4h), np.nan)
    
    # HMA 21 for trend
    hma_21 = calculate_hma(close_4h, 21)
    
    # Align to lower timeframe
    dc_high_aligned = align_htf_to_ltf(prices, df_4h, dc_high)
    dc_low_aligned = align_htf_to_ltf(prices, df_4h, dc_low)
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # 4h ATR for volatility filter
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(dc_high_aligned[i]) or np.isnan(dc_low_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility
        if i >= 30:
            vol_filter = atr_4h_aligned[i] > np.percentile(atr_4h_aligned[:i+1], 25)
        else:
            vol_filter = True
        
        if position == 0:
            # Uptrend: price > HMA21
            if price > hma_21_aligned[i]:
                # Long: price breaks above Donchian high with volume
                if (price > dc_high_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < HMA21
            elif price < hma_21_aligned[i]:
                # Short: price breaks below Donchian low with volume
                if (price < dc_low_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or Donchian low break
            if price < hma_21_aligned[i] or price < dc_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or Donchian high break
            if price > hma_21_aligned[i] or price > dc_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0