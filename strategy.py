#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRStop
Hypothesis: Donchian channel breakouts on 12h timeframe with 1d EMA50 trend filter and volume spike capture major trend moves while avoiding whipsaws. Designed for 15-30 trades/year to minimize fee drag and work in both bull and bear markets by using trend filter and volatility-based stops.
"""

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
    
    # 12h Donchian channel (20 periods)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian bands
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe (already aligned, but for safety)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stoploss (14 periods)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and above EMA50
            if price > upper and vol_spike and price > ema_50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price breaks below lower Donchian with volume and below EMA50
            elif price < lower and vol_spike and price < ema_50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr_val
        
        elif position == 1:
            signals[i] = 0.25
            # Stoploss: price closes below ATR stop
            if price < atr_stop:
                signals[i] = 0.0
                position = 0
            # Exit: price closes below lower Donchian
            elif price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Stoploss: price closes above ATR stop
            if price > atr_stop:
                signals[i] = 0.0
                position = 0
            # Exit: price closes above upper Donchian
            elif price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0