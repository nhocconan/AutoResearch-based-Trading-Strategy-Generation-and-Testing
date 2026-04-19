#!/usr/bin/env python3
# 6h_SuperTrend_Filtered_Breakout
# Hypothesis: 6h SuperTrend (ATR=10, mult=3) defines trend direction, price breaks above/below
# previous day's ATR-based channels with volume confirmation. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year). Uses discrete sizing 0.25 to minimize churn.

name = "6h_SuperTrend_Filtered_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # SuperTrend calculation (ATR=10, multiplier=3)
    def supertrend(high, low, close, atr_period=10, multiplier=3):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.full_like(close, np.nan)
        if len(close) >= atr_period:
            atr[atr_period-1] = np.nanmean(tr[:atr_period])
            for i in range(atr_period, len(tr)):
                if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                    atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
                else:
                    atr[i] = np.nan
        
        # Basic Upper and Lower Bands
        basic_ub = (high + low) / 2 + multiplier * atr
        basic_lb = (high + low) / 2 - multiplier * atr
        
        # Final Upper and Lower Bands
        final_ub = np.full_like(close, np.nan)
        final_lb = np.full_like(close, np.nan)
        final_ub[0] = basic_ub[0]
        final_lb[0] = basic_lb[0]
        
        for i in range(1, len(close)):
            if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        # SuperTrend
        supertrend = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i == 0:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
        
        return supertrend, atr
    
    # 6h data for SuperTrend and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    st, atr = supertrend(df_6h['high'].values, df_6h['low'].values, df_6h['close'].values, 10, 3)
    st_aligned = align_htf_to_ltf(prices, df_6h, st)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    
    # Previous day's ATR-based channels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Previous day's ATR (14-period)
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.full_like(close, np.nan)
        if len(close) >= period:
            atr[period-1] = np.nanmean(tr[:period])
            for i in range(period, len(tr)):
                if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                    atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                else:
                    atr[i] = np.nan
        return atr
    
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_prev = atr_1d.shift(1).values  # Previous day's ATR
    
    # Upper and lower channels: previous day close ± 1.5 * previous day ATR
    upper_channel = pc + 1.5 * atr_1d_prev
    lower_channel = pc - 1.5 * atr_1d_prev
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(st_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SuperTrend bullish (price > SuperTrend) AND price breaks above upper channel with volume
            if (close[i] > st_aligned[i] and 
                close[i] > upper_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: SuperTrend bearish (price < SuperTrend) AND price breaks below lower channel with volume
            elif (close[i] < st_aligned[i] and 
                  close[i] < lower_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below SuperTrend (trend reversal) OR price breaks below lower channel
            if (close[i] < st_aligned[i]) or (close[i] < lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above SuperTrend (trend reversal) OR price breaks above upper channel
            if (close[i] > st_aligned[i]) or (close[i] > upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals