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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily True Range and ATR (14-period)
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
    
    # Calculate daily ATR percentage (ATR / Close) for volatility filter
    atr_pct_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d[i]) and close_1d[i] > 0:
            atr_pct_1d[i] = atr_1d[i] / close_1d[i]
        else:
            atr_pct_1d[i] = 0.0
    
    # Calculate daily volume ratio (current volume / 20-period average volume)
    vol_ratio_1d = np.zeros(len(df_1d))
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            vol_avg = np.mean(volume_1d[i-19:i+1])
            if vol_avg > 0:
                vol_ratio_1d[i] = volume_1d[i] / vol_avg
            else:
                vol_ratio_1d[i] = 1.0
    
    # Align indicators to 4h timeframe (primary timeframe)
    atr_pct_4h = align_htf_to_ltf(prices, df_1d, atr_pct_1d)
    vol_ratio_4h = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
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
        if (np.isnan(atr_pct_4h[i]) or
            np.isnan(vol_ratio_4h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR% > 0.015 (1.5%)
        # Volume filter: volume ratio > 1.5 (50% above average)
        if atr_pct_4h[i] <= 0.015 or vol_ratio_4h[i] <= 1.5:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volatility and volume confirmation
            if close[i] > donch_high[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low with volatility and volume confirmation
            elif close[i] < donch_low[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_VolVol_Breakout"
timeframe = "4h"
leverage = 1.0