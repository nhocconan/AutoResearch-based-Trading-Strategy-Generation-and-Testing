#!/usr/bin/env python3
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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility filter (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily EMA50 for trend filter (1d)
    ema50_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(df_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h Donchian channels (20-period) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or
            np.isnan(ema50_12h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_12h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        # Calculate daily pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Camarilla-style pivot levels (R4/S4)
        r4 = prev_close + (prev_range * 1.1 / 2)
        s4 = prev_close - (prev_range * 1.1 / 2)
        
        # Align to 4h timeframe (no extra delay needed for daily pivot)
        r4_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r4))[i]
        s4_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s4))[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume confirmation and above daily EMA50
            if close[i] > donch_high[i] and volume_ratio > vol_threshold and close[i] > ema50_12h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low with volume confirmation and below daily EMA50
            elif close[i] < donch_low[i] and volume_ratio > vol_threshold and close[i] < ema50_12h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below daily EMA50
            if close[i] < donch_low[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above daily EMA50
            if close[i] > donch_high[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_EMA50_Volume"
timeframe = "4h"
leverage = 1.0