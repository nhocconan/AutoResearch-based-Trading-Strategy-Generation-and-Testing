#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 20-period Donchian channel breakout with 1-day EMA trend filter and volume confirmation.
Breakouts occur when price breaks above/below Donchian channels, filtered by daily EMA(50) to ensure trend alignment.
Volume > 1.5x average confirms breakout strength. Uses discrete positions (±0.25) and ATR-based stoploss.
Designed to work in bull/bear by capturing strong trending moves with volatility expansion.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 4h timeframe (waits for 1d bar close)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    donch_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donch_period-1, n):
        upper_channel[i] = np.max(high[i-donch_period+1:i+1])
        lower_channel[i] = np.min(low[i-donch_period+1:i+1])
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Donchian (20), ATR (14), EMA (50), volume MA (20)
    start_idx = max(donch_period-1, atr_period, ema_period*2, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below daily EMA(50)
        price_above_ema = price > ema_aligned[i]
        price_below_ema = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above upper channel in uptrend with volume
            if price_above_ema and price > upper_channel[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower channel in downtrend with volume
            elif price_below_ema and price < lower_channel[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower channel or trend reverses or stoploss
            if price < lower_channel[i] or not price_above_ema or price < (upper_channel[i] - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper channel or trend reverses or stoploss
            if price > upper_channel[i] or not price_below_ema or price > (lower_channel[i] + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_EMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0