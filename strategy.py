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
    
    # Get weekly data for calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly EMA14 for trend filter
    close_1w = df_1w['close'].values
    ema_14_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        ema_14_1w[13] = np.mean(close_1w[:14])
        for i in range(14, len(close_1w)):
            ema_14_1w[i] = (close_1w[i] * 2 + ema_14_1w[i-1] * 12) / 14  # EMA14
    
    # Calculate previous week's OHLC for Donchian channels (avoid look-ahead)
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(df_1w['high'].values, 1)
    prev_low = np.roll(df_1w['low'].values, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Weekly Donchian(14) channels
    upper_channel = np.maximum.accumulate(prev_high)
    lower_channel = np.minimum.accumulate(prev_low)
    
    # Align weekly indicators to daily timeframe
    ema_14_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_14_1w)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    
    # Calculate daily ATR(14) for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_14_1w_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above weekly upper channel with volume and above weekly EMA14
            if price > upper_channel_aligned[i] and vol_filter and price > ema_14_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly lower channel with volume and below weekly EMA14
            elif price < lower_channel_aligned[i] and vol_filter and price < ema_14_1w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly lower channel or trailing stop
            if price < lower_channel_aligned[i] or price < ema_14_1w_aligned[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly upper channel or trailing stop
            if price > upper_channel_aligned[i] or price > ema_14_1w_aligned[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian14_1wEMA14_Volume"
timeframe = "1d"
leverage = 1.0