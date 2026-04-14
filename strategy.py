#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR and range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for daily ATR
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) daily
    atr_14_1d = np.full_like(close_1d, np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.nanmean(tr[1:15])  # Skip first NaN
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Range ratio: daily range / ATR(14)
    range_ratio = np.full_like(close_1d, np.nan)
    mask = (atr_14_1d > 0) & ~np.isnan(atr_14_1d)
    range_ratio[mask] = daily_range[mask] / atr_14_1d[mask]
    
    # Align to 4h timeframe
    atr_14_1d_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    range_ratio_4h = align_htf_to_ltf(prices, df_1d, range_ratio)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # EMA(50) weekly for trend
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 0.0377 + ema_50_1w[i-1] * 0.9623)  # alpha = 2/(50+1)
    
    ema_50_1w_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_1d_4h[i]) or 
            np.isnan(range_ratio_4h[i]) or 
            np.isnan(ema_50_1w_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Low volatility (range ratio < 0.8) + price above weekly EMA50
            if (range_ratio_4h[i] < 0.8 and 
                close[i] > ema_50_1w_4h[i]):
                position = 1
                signals[i] = position_size
            # Short: Low volatility (range ratio < 0.8) + price below weekly EMA50
            elif (range_ratio_4h[i] < 0.8 and 
                  close[i] < ema_50_1w_4h[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: High volatility (range ratio > 1.2) OR price crosses below weekly EMA50
            if (range_ratio_4h[i] > 1.2 or 
                close[i] < ema_50_1w_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: High volatility (range ratio > 1.2) OR price crosses above weekly EMA50
            if (range_ratio_4h[i] > 1.2 or 
                close[i] > ema_50_1w_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_ATR_RangeRatio_1w_EMA50"
timeframe = "4h"
leverage = 1.0