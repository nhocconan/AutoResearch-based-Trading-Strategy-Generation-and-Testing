#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter, volume confirmation, and ATR stoploss
# Uses Donchian channel breakouts for trend following, filtered by 12h EMA direction and volume spikes.
# Works in both bull and bear markets by only taking breakouts in the direction of the higher timeframe trend.
# Target: 20-40 trades/year to minimize fee decay while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 50-period EMA on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_len = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_12h[ema_len-1] = np.mean(close_12h[:ema_len])
        for i in range(ema_len, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Calculate average volume on 4h for spike detection
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(vol_4h), np.nan)
    vol_period = 10
    for i in range(vol_period, len(vol_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-vol_period:i])
    
    # Calculate ATR(14) on 4h for stoploss
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr_4h = np.full(len(tr), np.nan)
    atr_period = 14
    for i in range(atr_period, len(tr)):
        if not np.isnan(tr[i-atr_period+1:i+1]).any():
            atr_4h[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(30, 50) + 10
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h_aligned[i] if vol_ma_4h_aligned[i] > 0 else 0
        
        # Volume confirmation: at least 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: Donchian breakout above upper band with uptrend and volume
            if price > donchian_high_aligned[i] and price > ema_12h_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Donchian breakout below lower band with downtrend and volume
            elif price < donchian_low_aligned[i] and price < ema_12h_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian lower band or trend reverses
            if price < donchian_low_aligned[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian upper band or trend reverses
            if price > donchian_high_aligned[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0