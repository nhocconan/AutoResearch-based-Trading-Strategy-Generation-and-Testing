#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    12h Channels: Donchian(20) breakout with weekly trend and volume confirmation.
    Long: Break above 20-period high + weekly EMA20 uptrend + volume > 1.5x avg
    Short: Break below 20-period low + weekly EMA20 downtrend + volume > 1.5x avg
    Exit: Opposite Donchian break or trailing stop at 2x ATR
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20  # EMA20
    
    # Align weekly EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 12h ATR(14) for volatility and stop loss
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
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Break above Donchian high with volume and weekly uptrend
            if price > donchian_high[i] and vol_filter and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = size
                position = 1
            # Short: Break below Donchian low with volume and weekly downtrend
            elif price < donchian_low[i] and vol_filter and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Break below Donchian low or trailing stop
            if price < donchian_low[i] or price < ema_20_1w_aligned[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Break above Donchian high or trailing stop
            if price > donchian_high[i] or price > ema_20_1w_aligned[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_WeeklyEMA20_Trend_Volume"
timeframe = "12h"
leverage = 1.0