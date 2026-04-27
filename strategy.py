#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_20 = np.full(len(close_1w), np.nan)
    period = 20
    alpha = 2 / (period + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_20[i] = close_1w[i]
        elif np.isnan(ema_20[i-1]):
            ema_20[i] = close_1w[i]
        else:
            ema_20[i] = alpha * close_1w[i] + (1 - alpha) * ema_20[i-1]
    
    # Align weekly EMA to 6h
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Calculate 14-period ATR for volatility and stop
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 10-period high/low for Donchian breakout
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 10
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and price above weekly EMA20
            if price > high_max[i] and vol_ratio > 1.5 and price > ema_20_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and price below weekly EMA20
            elif price < low_min[i] and vol_ratio > 1.5 and price < ema_20_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 1.5x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 1.5x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian10_EMA20_Trend_Volume_ATRStop_v1"
timeframe = "6h"
leverage = 1.0