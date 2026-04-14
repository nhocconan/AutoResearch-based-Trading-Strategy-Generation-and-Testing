#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Hypothesis: 6h Donchian breakout with 12h/1d trend filter and volume confirmation
    captures momentum in both bull and bear markets. Trend filter ensures we only
    trade in direction of higher timeframe trend, reducing whipsaw. Volume ensures
    breakout legitimacy. Target: 50-150 trades over 4 years.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data (HTF) once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(df_12h)):
            ema50_12h[i] = (close_12h[i] + ema50_12h[i-1]) * 0.5  # EMA(50) approx
    
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    atr_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 14:
        atr_12h[13] = np.mean(tr_12h[:14])
        for i in range(14, len(df_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate 6h Donchian channels (20-period)
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
        if (np.isnan(ema50_12h_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_12h_aligned[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation and above 12h EMA50
            if close[i] > donch_high[i] and volume_ratio > vol_threshold and close[i] > ema50_12h_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low with volume confirmation and below 12h EMA50
            elif close[i] < donch_low[i] and volume_ratio > vol_threshold and close[i] < ema50_12h_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below Donchian low OR below 12h EMA50
            if close[i] < donch_low[i] or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above Donchian high OR above 12h EMA50
            if close[i] > donch_high[i] or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Donchian_EMA50_Volume"
timeframe = "6h"
leverage = 1.0