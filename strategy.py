#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 12h Supertrend for trend direction and 12h Donchian(20) breakout
    # for entry timing, filtered by 1d volume confirmation. Uses discrete position sizing (0.25)
    # to minimize fee churn. Designed for low trade frequency (target: 12-37/year) to overcome
    # the 6h timeframe's historically poor keep rate by combining strong trend filter with
    # structural breakouts and volume confirmation - proven effective in both bull and bear markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate ATR for Supertrend (10-period)
    atr_period = 10
    tr = np.maximum(np.maximum(high_12h[1:] - low_12h[1:], 
                               np.abs(high_12h[1:] - close_12h[:-1])),
                    np.abs(low_12h[1:] - close_12h[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr = np.full(len(df_12h), np.nan)
    for i in range(atr_period, len(df_12h)):
        if np.isnan(atr[i-1]):
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    factor = 3.0
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    supertrend = np.full(len(df_12h), np.nan)
    direction = np.full(len(df_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_12h)):
        if np.isnan(supertrend[i-1]):
            # Initialize
            supertrend[i] = lowerband[i]
            direction[i] = 1
        else:
            if close_12h[i] > upperband[i-1]:
                direction[i] = 1
            elif close_12h[i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high_12h = np.full(len(df_12h), np.nan)
    donchian_low_12h = np.full(len(df_12h), np.nan)
    
    for i in range(20, len(df_12h)):
        donchian_high_12h[i] = np.max(high_12h[i-20:i])
        donchian_low_12h[i] = np.min(low_12h[i-20:i])
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA (20-period) for confirmation
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Volume confirmation: volume > 1.5 * 20-period average (1d)
    volume_spike = volume_1d > (1.5 * vol_ma_1d)
    
    # Align all HTF indicators to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(direction_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend from Supertrend direction
        bullish_trend = direction_aligned[i] == 1
        bearish_trend = direction_aligned[i] == -1
        
        # Entry logic: Donchian breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Donchian high in bullish trend with volume
        if bullish_trend:
            long_entry = (close[i] > donchian_high_aligned[i]) and volume_spike_aligned[i]
        # Short breakout: price breaks below Donchian low in bearish trend with volume
        elif bearish_trend:
            short_entry = (close[i] < donchian_low_aligned[i]) and volume_spike_aligned[i]
        
        # Exit logic: opposite Donchian level or trend reversal
        long_exit = bearish_trend and close[i] < donchian_low_aligned[i]
        short_exit = bullish_trend and close[i] > donchian_high_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_supertrend_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0