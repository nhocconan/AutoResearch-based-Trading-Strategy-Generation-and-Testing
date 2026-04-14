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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period) with vectorized implementation
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily EMA (50-period)
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(df_1d)):
            ema_50_1d[i] = (close_1d[i] * multiplier) + (ema_50_1d[i-1] * (1 - multiplier))
    
    # Calculate daily volatility filter (ATR > 0.8% of price)
    vol_filter_1d = np.zeros(len(df_1d), dtype=bool)
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d[i]) and close_1d[i] > 0:
            vol_filter_1d[i] = atr_1d[i] / close_1d[i] > 0.008
    
    # Calculate daily volume average (20-period)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        vol_ma_1d[19] = np.mean(vol_1d[:20])
        for i in range(20, len(df_1d)):
            vol_ma_1d[i] = (vol_ma_1d[i-1] * 19 + vol_1d[i]) / 20
    
    # Calculate volume spike filter (current volume > 1.5x 20-day average)
    vol_spike_1d = np.zeros(len(df_1d), dtype=bool)
    for i in range(len(df_1d)):
        if not np.isnan(vol_ma_1d[i]) and vol_ma_1d[i] > 0:
            vol_spike_1d[i] = vol_1d[i] > vol_ma_1d[i] * 1.5
    
    # Align indicators to 4h timeframe (primary timeframe)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_filter_4h = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float))
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 4-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_4h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema_50_4h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.8% of price)
        if vol_filter_4h[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # Calculate daily pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Camarilla-style pivot levels (R4/S4)
        r4 = prev_close + (prev_range * 1.1 / 2)
        s4 = prev_close - (prev_range * 1.1 / 2)
        
        # Align to 4h timeframe
        r4_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r4))[i]
        s4_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s4))[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high AND above S4 AND price > daily EMA50 AND volume spike
            if close[i] > donch_high[i] and close[i] > s4_4h and close[i] > ema_50_4h[i] and vol_spike_4h[i] > 0.5:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low AND below R4 AND price < daily EMA50 AND volume spike
            elif close[i] < donch_low[i] and close[i] < r4_4h and close[i] < ema_50_4h[i] and vol_spike_4h[i] > 0.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below S4 OR price < daily EMA50
            if close[i] < donch_low[i] or close[i] < s4_4h or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above R4 OR price > daily EMA50
            if close[i] > donch_high[i] or close[i] > r4_4h or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_R4S4_EMA50_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0