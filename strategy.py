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
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period) - Wilder's smoothing
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    low_close = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to daily timeframe
    atr_daily = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_daily[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_daily[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 70% of 20-period MA)
        if volume[i] < 0.7 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high AND volatility expansion
            if close[i] > donch_high[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low AND volatility expansion
            elif close[i] < donch_low[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR volatility contraction
            if close[i] < donch_low[i] or atr_daily[i] / close[i] < 0.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR volatility contraction
            if close[i] > donch_high[i] or atr_daily[i] / close[i] < 0.002:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_ATR_Donchian20_VolumeFilter"
timeframe = "1d"
leverage = 1.0