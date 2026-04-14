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
    
    # Load 1d data for reference levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR on 1d for volatility filter
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - high[i-1]), 
                       abs(low[i] - low[i-1]))
        
        atr = np.full_like(high, np.nan)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period EMA on 1d close for trend filter
    def calculate_ema(close, period):
        ema = np.full_like(close, np.nan)
        if len(close) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(close[:period])
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema20_1d = calculate_ema(close_1d, 20)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate volume spike detector on 1d
    def calculate_volume_ratio(volume, period=20):
        vol_ma = np.full_like(volume, np.nan)
        for i in range(period-1, len(volume)):
            vol_ma[i] = np.mean(volume[i-period+1:i+1])
        vol_ratio = np.zeros_like(volume)
        for i in range(len(volume)):
            if vol_ma[i] > 0:
                vol_ratio[i] = volume[i] / vol_ma[i]
            else:
                vol_ratio[i] = 0
        return vol_ratio
    
    vol_ratio_1d = calculate_volume_ratio(volume_1d, 20)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate dynamic thresholds based on volatility
        atr_multiple = atr_1d_aligned[i] * 0.5
        
        if position == 0:
            # Long: price above EMA20, breaking above recent high with volume spike
            if (close[i] > ema20_1d_aligned[i] and
                close[i] > high_1d_aligned[i] + atr_multiple and
                vol_ratio_1d_aligned[i] > 2.0):
                position = 1
                signals[i] = position_size
            # Short: price below EMA20, breaking below recent low with volume spike
            elif (close[i] < ema20_1d_aligned[i] and
                  close[i] < low_1d_aligned[i] - atr_multiple and
                  vol_ratio_1d_aligned[i] > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA20 or volatility drops
            if (close[i] < ema20_1d_aligned[i] or
                vol_ratio_1d_aligned[i] < 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA20 or volatility drops
            if (close[i] > ema20_1d_aligned[i] or
                vol_ratio_1d_aligned[i] < 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_EMA_Volatility_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0