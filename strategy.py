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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for volatility
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if i == 14:
            atr_1w[i] = np.mean(tr_1w[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Calculate weekly Donchian channels (20-period)
    upper_20 = np.full(len(close_1w), np.nan)
    lower_20 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        upper_20[i] = np.max(high_1w[i-20:i])
        lower_20[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50  # EMA50
    
    # Align weekly indicators to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily ATR(10) for stoploss
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(10, n):
        if i == 10:
            atr[i] = np.mean(tr[1:11])
        else:
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 20, 10, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above weekly upper Donchian with volume and above weekly EMA50
            if price > upper_20_aligned[i] and vol_filter and price > ema_50_1w_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly lower Donchian with volume and below weekly EMA50
            elif price < lower_20_aligned[i] and vol_filter and price < ema_50_1w_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly lower Donchian or trailing stop
            if price < lower_20_aligned[i] or price < ema_50_1w_aligned[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly upper Donchian or trailing stop
            if price > upper_20_aligned[i] or price > ema_50_1w_aligned[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0