#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price channel breakout (Donchian 20) with 1-day ATR filter and volume confirmation.
# Donchian breakouts capture momentum bursts. ATR filter ensures volatility is sufficient to avoid false breakouts.
# Volume confirms institutional participation. Works in bull markets (breakouts continue) and bear markets 
# (breakdowns from strength). Target: 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for ATR filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation on daily data (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h data for Donchian breakout and volume ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = np.zeros_like(high_4h)
    donchian_low = np.zeros_like(low_4h)
    
    for i in range(len(high_4h)):
        if i >= 19:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_avg20_4h = np.zeros_like(volume_4h)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_avg20_4h[i] = np.mean(volume_4h[i-19:i+1])
        else:
            vol_avg20_4h[i] = np.nan
    
    vol_avg20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg20_4h)
    
    signals = np.zeros(n)
    position = 0
    warmup = 100  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_avg20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_filter = vol_4h_current > 1.5 * vol_avg20_4h_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + sufficient volatility (ATR > 0.5% of price) + volume
            if close[i] > donchian_high_aligned[i] and atr_1d_aligned[i] > 0.005 * close[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + sufficient volatility + volume
            elif close[i] < donchian_low_aligned[i] and atr_1d_aligned[i] > 0.005 * close[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0